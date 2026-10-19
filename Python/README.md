# SubDisc in Python

This folder is my Python port of SubDisc (Subgroup Discovery, the tool that used to be called Cortana). The Java desktop application lives in **../Java** and I kept it there as is. The goal of the port: same subgroups, same qualities and same result files as the Java version, with a Qt user interface on top.

## What is needed

- Python 3.10 or newer (I run it on 3.14, global interpreter, no virtual environment)
- numpy
- PySide6 (the GUI)
- matplotlib (the charts in the GUI)
- pytest, only for the tests
- pymysql, only for the hidden "Load PSR Table" menu item

One time install:

**pip install numpy PySide6 matplotlib pytest**

## Starting it

From this folder:

- **python -m subdisc** opens the main window and asks for a data file, like the Java jar does
- **python SubDisc.py** does the same thing
- **python -m subdisc experiments.xml** runs every experiment in an autorun file and writes the result files next to it, no window at all (so this also works on a machine without PySide6)
- **python -m subdisc experiments.xml true** runs them and keeps the result windows open
- a third argument is the number of threads, it is accepted for compatibility but the port always mines on one thread

Data files are the same as for Java: .txt / .csv (comma, tab or semicolon), .arff and the .xml autorun files the GUI writes with "Create Autorun File".

## Layout

- **subdisc/** the core, one module per Java class. File names are snake_case (**SubgroupDiscovery.java** is **subgroup_discovery.py**), the class names stay the same and methods go from camelCase to snake_case. Three modules got another name because Python has a standard module with that name: **operators.py**, **stat_enum.py** and **column_statistics.py**
- **subdisc/java_compat.py** the small things that make the numbers come out like in Java: float32 arithmetic, Float.toString / Double.toString, java.util.Random, %f formatting and the like
- **subdisc/gui/** the windows, one module per Java window (MiningWindow, ResultWindow, BrowseWindow, MetaDataWindow, HistogramWindow, ModelWindow, ROC curve, the dialogs and so on)
- **subdisc/ui_hooks.py** the places where the Java core pops up a dialog. The core calls these hooks and the GUI fills them in, so the core does not import Qt
- **tests/** the Java JUnit tests ported to pytest, plus a small GUI test that runs offscreen

## The GUI

It is PySide6 with a flat light look close to the FlatLaf IntelliJ theme the Java version uses (there is a dark one under View). The windows, buttons, menus, keyboard shortcuts and the order in which the fields update are the same as in Java. Things I changed on purpose:

- mining, loading a file, cross-validation and computing random qualities run on a background thread with a small progress dialog. The Stop button (or Escape) ends a mining run and shows the result so far, same as in Java
- the progress line ("d=2 cands=1.234 evaluating: ...") is in the title bar like in Java and also in the status bar
- the charts are matplotlib, so zoom, pan and save as image come with the toolbar
- the ROC window computes the quality isometrics one time and keeps them, resizing is instant

## How I checked it is the same

I compiled the Java source from **../Java/src** (the jar in **../Java/target** is older than the source, revision 1140 against 1161, and it gives different results on data with missing values) and ran the same settings through Java and through the port, then compared the result files byte for byte. Identical for:

- single nominal: 19 settings, all search strategies (beam, ROC beam, cover based beam selection, best / depth / breadth first), all numeric strategies, value sets, intervals, minimum support and filtering
- single numeric: 12 settings, also with missing values
- double correlation 7, double regression 3, double binary 2, SCaPE 2
- multi-label 2 (with the same fixed random seed on both sides)
- label ranking 7, multi-numeric 4

On my machine the port is about 1 to 5 times slower than Java.

The tests: **python -m pytest tests** (they read the data files from **../Java/src/test/resources**).

## Where it differs from Java

On purpose, and only where Java crashes, hangs or the GUI loses what the user typed:

- the search runs on one thread. Java with 1 thread gives the same result as Java with more threads, and that is what I compared against
- the propensity score measures use a logistic regression. Java uses Weka for that; here it is a plain maximum likelihood fit with numpy, so the last digits can differ. These measures are hidden in the Java GUI anyway ("not implemented")
- Validation could loop forever in Java (a nextSetBit(k) that never moves on), fixed
- crashes I avoided: lines with too few fields in a .txt file (now missing values), changing a column type in some corner cases, a null ConditionBase in the external knowledge loader
- the loaders close the columns at the end. Java keeps the grown arrays (its loops only look at the row count), numpy needs the exact length. Results are the same
- main window: selecting a subset or changing the meta data no longer resets the target type, target and quality measure. Loading an xml file now also restores minimum support, filter subgroups and minimum improvement, and the target value of the right attribute
- the subset dialog gives binary columns a 0 / 1 choice (Java shows an empty box there)
- "Remove Enrichment Source" removes the selected domains (the Java version counts the wrong thing and removes at most the first one)
- the label ranking base model window shows the base matrix (Java builds it but never adds it to the window)
- Cross-Validate: Java hands the fold selection (row numbers of the full table) to the run together with the fold table, which already holds only the training rows, so the row numbers do not line up. Here the fold table is mined without that selection. And every fold gets its own copy of the settings, so the buttons in a fold's result window (p-values, regression test) work on that fold

Still the same as Java, even if it looks odd:

- "Show Model" for multi-numeric writes the three .dat grids plus gnuplot scripts into the current folder every time
- the SCaPE negatives stop at the last positive row (BitSet.flip up to length())
- "Resample" in the scatter plot draws the same sample again

## Not ported

Code that is dead or broken in the Java source as well:

- Miki, ConditionBuilder, DistributionBuilder, MultiSort, RankComparator, Histogram, DAGWindow
- RefinementList, Filter / Fltr and the deprecated mine() variants
- the **postprocess** package, a separate tool with hard coded paths of the original author's machine and unfinished parts
- Jama (numpy does the matrix work), FileTypeFilter and OnlyExt (Qt file dialogs have their own filters)
- the KNIME glue
