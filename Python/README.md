# SubDisc in Python

This folder is our Python port of SubDisc (Subgroup Discovery, the tool that used to be called Cortana). The Java desktop application lives in **../Java** and We kept it there as is. The goal of the port: same subgroups, same qualities and same result files as the Java version, with a Qt user interface on top.

## What is needed

- Python 3.10 or newer (We run it on 3.14, global interpreter, no virtual environment)
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

It is PySide6 with a flat light look close to the FlatLaf IntelliJ theme the Java version uses (there is a dark one under View). The windows, buttons, menus, keyboard shortcuts and the order in which the fields update are the same as in Java. Things We changed on purpose:

- mining, loading a file, cross-validation and computing random qualities run on a background thread with a small progress dialog. The Stop button (or Escape) ends a mining run and shows the result so far, same as in Java
- the progress line ("d=2 cands=1.234 evaluating: ...") is in the title bar like in Java and also in the status bar
- the charts are matplotlib, so zoom, pan and save as image come with the toolbar
- the ROC window computes the quality isometrics one time and keeps them, resizing is instant
- every table with column headings can be sorted and filtered (result window, pattern team tabs, browse, meta data, the ROC table, the cross table). A click on a heading sorts on that column: ascending, then descending, then back to the original order. The small arrow at the right end of a heading opens a filter with a value box and =, >, <, >=, <=, include (the cell contains the text) or exclude (it does not). Numbers are compared as numbers, text ignores upper / lower case, and filters on more columns must all hold
- a column filter can have a second condition, joined with and / or, e.g. Coverage >= 100 and <= 500, or Conditions include age or include sex. The second one only counts when its value box is not empty
- the filters of a table can be saved as a named preset (Presets in the filter box, or a right click on any heading) and applied again later in a window of the same kind, also after a restart. Columns are matched by their heading, a preset column the table does not have is reported. The presets are kept on this computer only (the same place as the light / dark choice, under HKCU\Software\SubDisc on Windows); result and autorun files stay as Java writes them
- in a ROC window the table filter also changes the picture: it is applied to all subgroups of that window, and the curve, the points and the area under the curve are drawn again from the ones that pass. The title says how many that are
- in the result window the filter carries over: ROC, Pattern Team, Save, Print, the p-value buttons and the regression test only use the subgroups that are still shown, and a line under the table says how many of them that is. Sorting changes the view only, the 'Nr.' of a subgroup stays the same
- in the browse window Save writes only the rows the filter leaves

## How We checked that it is the same

We compiled the Java source from **../Java/src** (the jar in **../Java/target** is older than the source, revision 1140 against 1161, and it gives different results on data with missing values) and ran the same settings through Java and through the port, then compared the result files byte for byte. Identical for:

- single nominal: 19 settings, all search strategies (beam, ROC beam, cover based beam selection, best / depth / breadth first), all numeric strategies, value sets, intervals, minimum support and filtering
- single numeric: 12 settings, also with missing values
- double correlation 7, double regression 3, double binary 2, SCaPE 2
- multi-label 2 (with the same fixed random seed on both sides)
- label ranking 7, multi-numeric 4

On our machine the port is about 1 to 5 times slower than Java. This is partly because Java is a compiled language, whereas Python is an interpreted language.

The tests: **python -m pytest tests** (they read the data files from **../Java/src/test/resources**).

## Where it differs from Java

On purpose, and only where Java crashes, hangs or the GUI loses what the user typed:

- the search runs on one thread. Java with 1 thread gives the same result as Java with more threads, and that is what We compared against
- the propensity score measures use a logistic regression. Java uses Weka for that; here it is a plain maximum likelihood fit with numpy, so the last digits can differ. These measures are hidden in the Java GUI anyway ("not implemented")
- Validation could loop forever in Java (a nextSetBit(k) that never moves on), fixed
- crashes We avoided: lines with too few fields in a .txt file (now missing values), changing a column type in some corner cases, a null ConditionBase in the external knowledge loader
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
