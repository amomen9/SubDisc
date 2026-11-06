"""Model error slicing (not in the java version): where does a classifier go
wrong? A logistic regression is trained on the table with k-fold cross
validation, so every row gets a prediction from a model that did not see
it (an out-of-fold prediction). Then new columns are added:

    <prefix>score            predicted probability of the positive class
    <prefix>error            1 when the prediction (score >= threshold) is wrong
    <prefix>false_positive   predicted positive, really negative
    <prefix>false_negative   predicted negative, really positive

Mining with <prefix>error as the target (single nominal, value 1) finds
the subgroups where the model fails more than on average, e.g. a fairness
check on adult: is the error rate for sex = 'Female' different? The score
column also fits the existing SCAPE target type (binary label + score).

The model: features are standardised numeric columns (missing -> 0 after
standardising, plus a 0/1 'was missing' column), binary columns as 0/1 and
nominal columns one-hot (missing is a level of its own). Fit by Newton's
method on the L2 penalised log-likelihood

    l(w) = sum_i [y_i log s(x_i w) + (1 - y_i) log(1 - s(x_i w))] - lambda/2 |w|^2

with s(t) = 1 / (1 + e^-t); the intercept is not penalised.
"""

import numpy as np

from . import log
from .attribute_type import AttributeType
from .column import Column


def design_matrix(table, feature_columns, rows=None):
    """X (rows x p) and the names of its columns"""
    n = table.get_nr_rows()
    parts = []
    names = []
    for c in feature_columns:
        t = c.get_type()
        miss = np.asarray(c.missing_view()[:n], dtype=bool)
        if t == AttributeType.NUMERIC:
            f = np.asarray(c.floats_view()[:n], dtype=np.float64).copy()
            bad = miss | np.isnan(f)
            ok = ~bad
            mu = f[ok].mean() if ok.any() else 0.0
            sd = f[ok].std() if ok.any() else 1.0
            sd = sd if sd > 0 else 1.0
            z = np.where(ok, (f - mu) / sd, 0.0)
            parts.append(z[:, None])
            names.append(c.get_name())
            if bad.any():
                parts.append(bad.astype(np.float64)[:, None])
                names.append(c.get_name() + " missing")
        elif t == AttributeType.BINARY:
            b = np.asarray(c.get_binaries()[:n], dtype=np.float64)
            parts.append(b[:, None])
            names.append(c.get_name())
        elif t == AttributeType.NOMINAL:
            codes = np.asarray(c.nominal_codes_view()[:n]).copy()
            values = list(c.distinct_values())
            codes = np.where(miss, len(values), codes)
            levels = values + ["(missing)"]
            used = np.unique(codes)
            # leave the most frequent level out, it is the intercept
            counts = np.bincount(codes, minlength=len(levels))
            ref = int(np.argmax(counts))
            for lv in used:
                if lv == ref:
                    continue
                parts.append((codes == lv).astype(np.float64)[:, None])
                names.append("%s = %s" % (c.get_name(), levels[lv]))
    X = np.hstack(parts) if parts else np.zeros((n, 0))
    return X, names


def fit_logistic(X, y, l2=1.0, max_iter=50, tol=1e-8):
    """weights (intercept first) of an L2 penalised logistic regression"""
    n, p = X.shape
    Xb = np.hstack((np.ones((n, 1)), X))
    w = np.zeros(p + 1)
    prior = min(max(y.mean(), 1e-6), 1 - 1e-6)
    w[0] = np.log(prior / (1 - prior))
    pen = np.full(p + 1, l2)
    pen[0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(Xb @ w, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        g = Xb.T @ (y - mu) - pen * w
        wts = mu * (1 - mu)
        H = (Xb * wts[:, None]).T @ Xb + np.diag(pen) + 1e-9 * np.eye(p + 1)
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < tol:
            break
    return w


def predict_logistic(w, X):
    eta = np.clip(w[0] + X @ w[1:], -35, 35)
    return 1.0 / (1.0 + np.exp(-eta))


def out_of_fold_scores(X, y, k=5, l2=1.0, seed=1):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    fold = np.empty(n, dtype=np.int64)
    perm = rng.permutation(n)
    fold[perm] = np.arange(n) % k
    scores = np.empty(n)
    for f in range(k):
        test = fold == f
        w = fit_logistic(X[~test], y[~test], l2)
        scores[test] = predict_logistic(w, X[test])
    return scores


def unique_name(table, name):
    existing = {c.get_name() for c in table.get_columns()}
    if name not in existing:
        return name
    i = 2
    while "%s_%d" % (name, i) in existing:
        i += 1
    return "%s_%d" % (name, i)


def add_model_error_columns(table, target_column, positive_value, feature_columns=None, k=5, l2=1.0,
                            threshold=0.5, seed=1, prefix="model_"):
    """trains the model, adds the four columns to table (in place) and
    returns (names of the new columns, summary dict)"""
    n = table.get_nr_rows()
    t = target_column.get_type()
    if t == AttributeType.BINARY:
        want = AttributeType.is_valid_binary_true_value(positive_value)
        y = np.asarray(target_column.get_binaries()[:n], dtype=bool) == want
    elif t == AttributeType.NOMINAL:
        values = list(target_column.distinct_values())
        if positive_value not in values:
            raise ValueError("'%s' is not a value of column '%s'" % (positive_value, target_column.get_name()))
        code = values.index(positive_value)
        y = np.asarray(target_column.nominal_codes_view()[:n]) == code
        y &= ~np.asarray(target_column.missing_view()[:n], dtype=bool)
    else:
        raise ValueError("the target of a classifier must be nominal or binary")
    if feature_columns is None:
        feature_columns = [c for c in table.get_columns() if c is not target_column and c.get_is_enabled()]
    feature_columns = [c for c in feature_columns if c is not target_column]
    if not feature_columns:
        raise ValueError("no feature columns")
    if y.all() or not y.any():
        raise ValueError("the target has only one class")
    X, names = design_matrix(table, feature_columns)
    log.log_command_line("model error columns: logistic regression on %d rows, %d features, %d folds" % (n, X.shape[1], k))
    scores = out_of_fold_scores(X, y.astype(np.float64), k, l2, seed)
    pred = scores >= threshold
    error = pred != y
    fp = pred & ~y
    fn = ~pred & y
    new = []
    for suffix, values, kind in (("score", scores.astype(np.float32), AttributeType.NUMERIC),
                                 ("error", error, AttributeType.BINARY),
                                 ("false_positive", fp, AttributeType.BINARY),
                                 ("false_negative", fn, AttributeType.BINARY)):
        name = unique_name(table, prefix + suffix)
        col = Column(name, None, kind, len(table.get_columns()), n)
        col.set_data(values)
        table.get_columns().append(col)
        new.append(name)
    table.update()
    from .subgroup_list import auc
    res = {
        "rows": n,
        "features (after one-hot)": X.shape[1],
        "folds": k,
        "accuracy": float(np.mean(~error)),
        "error rate": float(np.mean(error)),
        "false positives": int(np.count_nonzero(fp)),
        "false negatives": int(np.count_nonzero(fn)),
        "AUC (out of fold)": auc(scores, y),
    }
    return new, res
