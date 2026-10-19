"""Runs one experiment (java Process): picks the SubgroupDiscovery setup for
the target type, mines, and opens the result window when asked."""

import time

import numpy as np

from . import log
from . import ui_hooks
from .attribute_type import AttributeType
from .column import Column
from .qm import QM
from .subgroup_set import SubgroupSet
from .target_type import TargetType

# leave False
CAUC_LIGHT = False
CAUC_HEAVY = False
CAUC_HEAVY_CONVEX = False   # convex hull subgroups if True, top-1 if False
ROC_BEAM_TEST = False


def run_subgroup_discovery(table, fold, selection, search_parameters, show_windows, nr_threads, main_window=None):
    global CAUC_HEAVY
    from . import subgroup_discovery as sdm
    from .subgroup_discovery import SubgroupDiscovery
    tt = search_parameters.get_target_concept().get_target_type()
    if not TargetType.is_implemented(tt):
        return None

    if tt == TargetType.SINGLE_NOMINAL:
        # recompute, cross-validation or a selection may give a smaller number
        tc = search_parameters.get_target_concept()
        positives = tc.get_primary_target().count_values(tc.get_target_value(), selection)
        sd = SubgroupDiscovery.single_nominal(search_parameters, table, selection, positives, main_window)
    elif tt == TargetType.SINGLE_NUMERIC:
        if CAUC_HEAVY:
            _cauc_heavy(table, fold, selection, search_parameters, show_windows, nr_threads)
            return None
        elif sdm.TEMPORARY_CODE:
            _temporary_code(table, selection, search_parameters)
            return None
        avg = search_parameters.get_target_concept().get_primary_target().get_average(selection)
        from .java_compat import float_to_string
        log.log_command_line("average: " + float_to_string(avg))
        sd = SubgroupDiscovery.single_numeric(search_parameters, table, selection, avg, main_window)
    elif tt == TargetType.MULTI_NUMERIC:
        sd = SubgroupDiscovery.multi_numeric(table, selection, main_window, search_parameters)
    elif tt == TargetType.MULTI_LABEL:
        sd = SubgroupDiscovery.multi_label(search_parameters, table, selection, main_window)
    elif tt == TargetType.DOUBLE_REGRESSION:
        sd = SubgroupDiscovery.double_target(search_parameters, table, selection, True, main_window)
    elif tt in (TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY):
        # DOUBLE_BINARY is passed as if correlation
        sd = SubgroupDiscovery.double_target(search_parameters, table, selection, False, main_window)
    elif tt == TargetType.SCAPE:
        sd = SubgroupDiscovery.scape(main_window, search_parameters, table, selection)
    elif tt == TargetType.LABEL_RANKING:
        sd = SubgroupDiscovery.label_ranking(search_parameters, main_window, table, selection)
    else:
        raise AssertionError("Process: TargetType '%s' not implemented" % tt)

    begin = int(time.time() * 1000)
    sd.mine(int(time.time() * 1000), nr_threads)
    end = int(time.time() * 1000)
    max_time = search_parameters.get_maximum_time()
    if max_time > 0.0 and end > (begin + (max_time * 60000.0)):
        msg = "Mining process ended prematurely due to time limit."
        if show_windows:
            ui_hooks.show_message("Time Limit", msg)
        else:
            log.log_command_line(msg)

    # FIXME MM (java) for now hack this into the p-value column
    sd.get_result().mark_alternative_descriptions()

    if show_windows:
        # members are needed when the Table changes after this experiment
        for s in sd.get_result():
            s.revive_members()
        if ui_hooks.show_result_window is not None:
            ui_hooks.show_result_window(table, selection, sd)

    if CAUC_LIGHT:
        _cauc_light(sd, selection)
    if ROC_BEAM_TEST and tt == TargetType.SINGLE_NOMINAL:
        sd.get_result().get_roc_list()
    return sd


def echo_mining_end(milliseconds, nr_subgroups):
    minutes = milliseconds // 60000
    seconds = (milliseconds % 60000) / 1000.0
    s = "Mining process finished in %d minute%s and %.3f seconds.\n" % (minutes, "" if minutes == 1 else "s", seconds)
    if nr_subgroups == 0:
        s += "   No subgroups found that match the search criterion.\n"
    elif nr_subgroups == 1:
        s += "   1 subgroup found.\n"
    else:
        s += "   " + str(nr_subgroups) + " subgroups found.\n"
    log.log_command_line(s)


def echo_mining_start():
    log.log_command_line("Mining process started")


def _members_check(mask, size):
    return mask if mask is not None else np.ones(size, dtype=bool)


def _cauc_members(column, threshold, mask):
    f = column.floats_view()[:mask.shape[0]]
    mask &= ~(f > threshold)


def _cauc_light(sd, selection):
    target = sd.get_search_parameters().get_target_concept().get_primary_target()
    result = sd.get_result()
    members = _members_check(selection, target.size())
    dm = target.get_unique_numeric_domain_map(members, int(np.count_nonzero(members)))
    domain = dm.itsDomain[:dm.itsSize]
    statistics = []
    for i in range(len(domain) - 1):
        cs = members.copy()
        _cauc_members(target, domain[i], cs)
        result.set_binary_target(cs)
        statistics.append(_compile_statistics(float(domain[i]), int(np.count_nonzero(cs)), result))
    _cauc_write("caucLight", target, statistics)


def _cauc_heavy(table, fold, selection, sp, show_windows, nr_threads):
    global CAUC_HEAVY
    from .subgroup_set import SubgroupSet
    CAUC_HEAVY = False
    backup = sp.get_target_concept().get_primary_target().copy()
    members = _members_check(selection, backup.size())
    dm = backup.get_unique_numeric_domain_map(members, int(np.count_nonzero(members)))
    domain = dm.itsDomain[:dm.itsSize]
    columns = table.get_columns()
    name, short, index = backup.get_name(), backup.get_short(), backup.get_index()
    nr_rows = backup.get_index()   # sic
    tc = sp.get_target_concept()
    tc.set_target_type(TargetType.SINGLE_NOMINAL.GUI_TEXT)
    tc.set_target_value("1")
    backup_qm = sp.get_quality_measure()
    sp.set_quality_measure(QM.WRACC)   # XXX no motivation for this choice
    backup_mm = sp.get_quality_measure_minimum()
    sp.set_quality_measure_minimum(0.01)
    heavy = SubgroupSet.with_comparator(lambda x, y: x.get_conditions().compare_to(y.get_conditions()))
    for i in range(len(domain) - 1):
        cs = members.copy()
        _cauc_members(backup, domain[i], cs)
        col = Column(name, short, AttributeType.BINARY, index, nr_rows)
        for k in range(backup.size()):
            col.add_binary(bool(cs[k]))
        columns[index] = col
        sp.get_target_concept().set_primary_target(col)
        state = log.COMMANDLINELOG
        log.COMMANDLINELOG = False
        sd = run_subgroup_discovery(table, fold, members, sp, False, nr_threads, None)
        log.COMMANDLINELOG = state
        log.log_command_line("Threshold value : " + str(float(domain[i])))
        if CAUC_HEAVY_CONVEX:
            # java builds a (never shown) ROCCurve panel here, only for the
            # side effect of computing the ROC list
            sd.get_result().get_roc_list()
            roc = sd.get_result().get_roc_list_subgroup_set()
            for s in roc:
                log.log_command_line("    " + str(s.get_conditions()))
            heavy.add_all(roc)
        else:
            res = sd.get_result()
            if len(res) > 0:
                top = res.first()
                log.log_command_line("Subgroup : ")
                log.log_command_line("    " + str(top.get_conditions()))
                heavy.add(top)
    log.log_command_line("======================================================")
    log.log_command_line("Diverse Subgroup Set Size : " + str(len(heavy)))
    log.log_command_line("Subgroups : ")
    for s in heavy:
        log.log_command_line("    " + str(s.get_conditions()))
    state = log.COMMANDLINELOG
    log.COMMANDLINELOG = False
    with_entropy = heavy.get_pattern_team(table, len(heavy))
    log.COMMANDLINELOG = state
    log.log_command_line("Joint Entropy             : " + str(with_entropy.get_joint_entropy()))
    log.log_command_line("Entropy / Set Size        : " + str(with_entropy.get_joint_entropy() / max(1, len(heavy))))
    columns[index] = backup
    tc.set_primary_target(backup)
    tc.set_target_type(TargetType.SINGLE_NUMERIC.GUI_TEXT)
    sp.set_quality_measure(backup_qm)
    sp.set_quality_measure_minimum(backup_mm)
    CAUC_HEAVY = True


def _compile_statistics(threshold, nr_members, subgroup_set):
    """[threshold, n, AUC, fpr_1, tpr_1, ..., fpr_h, tpr_h]"""
    stats = [float(threshold), float(nr_members), subgroup_set.get_roc_list().get_area_under_curve(), 0.0, 0.0]
    for row in subgroup_set.get_roc_list_subgroups():
        stats.append(float(row[1]))
        stats.append(float(row[2]))
    stats.append(1.0)
    stats.append(1.0)
    return stats


def _cauc_write(caller, target, statistics):
    from .java_compat import double_to_string
    print("#" + caller)
    print("#" + target.get_name())
    print("#threshold,n,AUC,frp_1,tpr_1,...,fpr_h,tpr_h")
    for l in statistics:
        print("[" + ",".join(double_to_string(v) for v in l) + "]")
    if ui_hooks.show_cauc_window is not None:
        ui_hooks.show_cauc_window(target, statistics)


def _temporary_code(table, selection, sp):
    from . import subgroup_discovery as sdm
    from .subgroup_discovery import SubgroupDiscovery
    from .probability_density_function_nd import h_silverman
    from .xml_auto_run import save
    import math
    tc = sp.get_target_concept()
    target = tc.get_primary_target()
    h = h_silverman(1, table.get_nr_rows())
    mx = target.get_max()
    mn = target.get_min()
    rng = mx - mn
    nr_bins = int(math.ceil(rng / h))
    max_split = (4 * nr_bins) - 1
    print("\nrange / h = bins ( ceil(bins) )")
    print("%f / %f = %f ( %d )" % (rng, h, rng / h, nr_bins))
    path = "%s_%d" % (table.get_name(), time.perf_counter_ns())
    avg = target.get_average(selection)
    for i in range(1, max_split + 1):
        _temporary_code_helper(table, selection, sp, avg, i, False, path)
        _temporary_code_helper(table, selection, sp, avg, i, True, path)
    sdm.TEMPORARY_CODE = False
    sd = SubgroupDiscovery.single_numeric(sp, table, selection, avg, None)
    sd.mine(int(time.time() * 1000), 1)
    fn = "%s_%s.txt" % (path, "ORIG")
    print("\nwriting: " + fn)
    save(sd.get_result(), fn, sp.get_target_concept().get_target_type())
    sdm.TEMPORARY_CODE = True
    print("Done")


def _temporary_code_helper(table, selection, sp, avg, nr_split_points, use_equal_width, path):
    from . import subgroup_discovery as sdm
    from .subgroup_discovery import SubgroupDiscovery
    from .xml_auto_run import save
    sdm.TEMPORARY_CODE = True
    sdm.TEMPORARY_CODE_NR_SPLIT_POINTS = nr_split_points
    sdm.TEMPORARY_CODE_USE_EQUAL_WIDTH = use_equal_width
    kind = "EW" if use_equal_width else "EH"
    print("\nrunning: %s %d" % (kind, nr_split_points))
    sd = SubgroupDiscovery.single_numeric(sp, table, selection, avg, None)
    sd.mine(int(time.time() * 1000), 1)
    fn = "%s_%s_%05d.txt" % (path, kind, nr_split_points)
    print("\nwriting: " + fn)
    save(sd.get_result(), fn, sp.get_target_concept().get_target_type())
