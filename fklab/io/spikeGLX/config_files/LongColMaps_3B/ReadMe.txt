
===================
Long Column Pattern
===================

These maps are for 3B probes.

All even channels are bank 0; odd channels are bank 1.

This gives the effect of a single column extending across banks 0 and 1.

Specifically, the arrangement of bank selections in the included CheckPat_1shank.imro table is as follows:

Base
...
b0 b1
b0 b1
b0 b1
b0 b1
b0 b1
b0 b1
Tip

The included channel map: LongCol_1shank.imec.cmp sorts the graph pages as follows:

All the bank-0 AP,
All the bank-1 AP,
All the bank-0 LF,
All the bank-1 LF
Sync.

Apply these mappings using the 'Load' buttons in the imro and channel map editors, respectively.

Note that the ShankViewer will correctly display the activity.
