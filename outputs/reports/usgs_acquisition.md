# USGS large-sample acquisition (120 stations)

> **LEGACY / WITHDRAWN WORDING.** The 2019–2020 interval below informed station
> inclusion and later model/narrative development. It is an exploratory
> development-evaluation period, not a blind, untouched, or confirmatory test.
> The retained ledgers support an audit of the recorded decisions, but original
> discovery responses, timestamps, exact command line, and run configuration are
> unavailable, so the 1,465-candidate acquisition cannot be source-replayed.

_Window 2006-01-01…2020-12-31. Probed 1465 candidates in 12623s. Schema matches the original study._

## Inclusion criteria

- Full-record WTEMP coverage ≥ 0.55
- Full-record FLOW coverage ≥ 0.70
- Development-evaluation-window (2019-01-01–2020-12-31) WTEMP coverage ≥ 0.80
- Development-evaluation-window FLOW coverage ≥ 0.80

These thresholds ensured that accepted stations could both train the model on the pre-2019 record and contribute observations to the previously inspected 2019–2020 development evaluation. Every recorded candidate (kept or rejected) appears in `data_usgs/rejected_sites_120v2.csv` and `data_usgs/stations_meta_120v2.csv`; this makes the surviving ledger auditable but does not recreate the missing source requests.

## Kept stations

| site_id | USGS | state | name | WT cov | FL cov | WT cov 2019+ | FL cov 2019+ | WLEVEL cov |
|---|---|---|---|---|---|---|---|---|
| n00 | 14161500 | OR | LOOKOUT CREEK NEAR BLUE RIVER, OR | 0.581 | 1.0 | 1.0 | 1.0 | 0.0 |
| n01 | 14159500 | OR | SOUTH FORK MCKENZIE RIVER NEAR RAI | 0.999 | 1.0 | 1.0 | 1.0 | 0.0 |
| n02 | 07106000 | CO | FOUNTAIN CREEK NEAR FOUNTAIN, CO. | 0.862 | 1.0 | 0.841 | 1.0 | 0.0 |
| n03 | 07178200 | OK | Bird Ck at State Highway 266 near  | 0.909 | 1.0 | 0.984 | 1.0 | 0.0 |
| n04 | 08052700 | TX | Little Elm Ck nr Aubrey, TX | 0.666 | 1.0 | 0.953 | 1.0 | 0.998 |
| n05 | 09136100 | CO | NORTH FK GUNNISON RIVER ABOVE MOUT | 0.774 | 0.781 | 1.0 | 1.0 | 0.0 |
| n06 | 13342500 | ID | CLEARWATER RIVER AT SPALDING ID | 0.993 | 1.0 | 1.0 | 1.0 | 0.0 |
| n07 | 03077500 | PA | Youghiogheny River at Youghiogheny | 0.845 | 0.75 | 0.96 | 1.0 | 0.0 |
| n08 | 02334885 | GA | SUWANEE CREEK AT SUWANEE, GA | 0.988 | 1.0 | 0.995 | 1.0 | 0.985 |
| n09 | 12056500 | WA | NF SKOKOMISH R BL STAIRCASE RPDS N | 0.62 | 1.0 | 0.986 | 1.0 | 0.999 |
| n10 | 08065350 | TX | Trinity Rv nr Crockett, TX | 0.932 | 1.0 | 0.889 | 1.0 | 0.977 |
| n11 | 11507500 | OR | LINK RIVER AT KLAMATH FALLS, OR | 0.985 | 1.0 | 0.993 | 1.0 | 0.0 |
| n12 | 01426500 | NY | WEST BRANCH DELAWARE RIVER AT HALE | 0.997 | 1.0 | 1.0 | 1.0 | 0.0 |
| n13 | 14152000 | OR | MIDDLE FORK WILLAMETTE RIVER AT JA | 0.99 | 1.0 | 1.0 | 1.0 | 0.0 |
| n14 | 12210700 | WA | NOOKSACK RIVER AT NORTH CEDARVILLE | 0.835 | 1.0 | 1.0 | 1.0 | 0.999 |
| n15 | 09144250 | CO | GUNNISON RIVER AT DELTA, CO | 0.762 | 1.0 | 0.985 | 1.0 | 0.0 |
| n16 | 05054000 | ND | RED RIVER OF THE NORTH AT FARGO, N | 0.954 | 1.0 | 0.94 | 1.0 | 0.99 |
| n17 | 12398600 | WA | PEND OREILLE RIVER AT INTERNATIONA | 0.937 | 1.0 | 0.922 | 1.0 | 0.0 |
| n18 | 07332622 | TX | Bois D'Arc Ck at FM 409 nr Honey G | 0.68 | 0.772 | 0.978 | 1.0 | 0.761 |
| n19 | 04125460 | MI | PINE RIVER AT HIGH SCHOOL BRIDGE N | 0.985 | 1.0 | 0.967 | 1.0 | 0.0 |
| n20 | 02458450 | AL | VILLAGE CREEK AT AVENUE W AT ENSLE | 0.995 | 1.0 | 0.999 | 1.0 | 0.996 |
| n21 | 02110701 | SC | CRABTREE SWAMP AT CONWAY, SC | 0.927 | 0.839 | 0.996 | 0.952 | 0.369 |
| n22 | 14320934 | OR | LITTLE WOLF CREEK NEAR TYEE, OR | 0.866 | 0.889 | 0.943 | 1.0 | 0.0 |
| n23 | 02474560 | MS | LEAF RIVER NR NEW AUGUSTA, MS | 0.6 | 0.999 | 0.837 | 1.0 | 0.989 |
| n24 | 02336300 | GA | PEACHTREE CREEK AT ATLANTA, GA | 0.974 | 1.0 | 0.996 | 1.0 | 0.991 |
| n25 | 04027000 | WI | BAD RIVER NEAR ODANAH, WI | 0.613 | 1.0 | 0.971 | 1.0 | 0.0 |
| n26 | 14151000 | OR | FALL CREEK BLW WINBERRY CREEK, NEA | 0.987 | 1.0 | 1.0 | 1.0 | 0.0 |
| n27 | 01540500 | PA | Susquehanna River at Danville, PA | 0.62 | 1.0 | 0.969 | 1.0 | 0.0 |
| n28 | 02330450 | GA | CHATTAHOOCHEE RIVER AT HELEN, GA | 0.955 | 1.0 | 0.836 | 1.0 | 0.962 |
| n29 | 08070200 | TX | E Fk San Jacinto Rv nr New Caney,  | 0.989 | 0.981 | 0.938 | 1.0 | 0.974 |
| n30 | 03301900 | KY | FERN CREEK AT OLD BARDSTOWN RD AT  | 0.623 | 1.0 | 0.956 | 1.0 | 0.0 |
| n31 | 10351600 | NV | TRUCKEE RV BLW DERBY DAM NR WADSWO | 0.845 | 1.0 | 1.0 | 1.0 | 0.0 |
| n32 | 12117600 | WA | CEDAR RIVER BELOW DIVERSION NEAR L | 0.994 | 1.0 | 1.0 | 1.0 | 0.995 |
| n33 | 04043238 | MI | SALMON TROUT RIVER NEAR BIG BAY, M | 0.976 | 1.0 | 0.996 | 1.0 | 0.0 |
| n34 | 05057000 | ND | SHEYENNE RIVER NEAR COOPERSTOWN, N | 0.982 | 1.0 | 0.979 | 1.0 | 0.965 |
| n35 | 01104455 | MA | STONY BROOK, UNNAMED TRIBUTARY 1,  | 0.982 | 1.0 | 0.997 | 1.0 | 0.0 |
| n36 | 14211499 | OR | KELLEY CREEK AT SE 159TH DRIVE AT  | 0.966 | 1.0 | 0.997 | 1.0 | 0.0 |
| n37 | 12115000 | WA | CEDAR RIVER NEAR CEDAR FALLS, WA | 0.922 | 1.0 | 0.975 | 1.0 | 0.375 |
| n38 | 14153500 | OR | COAST FORK WILLAMETTE R BLW COTTAG | 0.997 | 1.0 | 0.992 | 1.0 | 0.0 |
| n39 | 01463500 | NJ | Delaware River at Trenton NJ | 0.957 | 1.0 | 0.97 | 1.0 | 0.0 |
| n40 | 04077630 | WI | RED RIVER AT MORGAN ROAD NEAR MORG | 0.991 | 1.0 | 1.0 | 1.0 | 0.0 |
| n41 | 01435000 | NY | NEVERSINK RIVER NEAR CLARYVILLE NY | 0.588 | 1.0 | 0.995 | 1.0 | 0.0 |
| n42 | 04137005 | MI | AU SABLE RIVER NEAR CURTISVILLE, M | 0.988 | 1.0 | 0.995 | 1.0 | 0.0 |
| n43 | 13018350 | WY | FLAT CREEK BELOW CACHE CREEK, NEAR | 0.707 | 1.0 | 0.964 | 1.0 | 0.0 |
| n44 | 03034000 | PA | Mahoning Creek at Punxsutawney, PA | 0.637 | 1.0 | 0.969 | 1.0 | 0.0 |
| n45 | 06894000 | MO | Little Blue River near Lake City,  | 0.987 | 1.0 | 0.985 | 1.0 | 0.0 |
| n46 | 09180000 | UT | DOLORES RIVER NEAR CISCO, UT | 0.947 | 1.0 | 0.989 | 1.0 | 0.0 |
| n47 | 01428500 | NY | DELAWARE R ABOVE LACKAWAXEN R NEAR | 0.986 | 1.0 | 0.956 | 1.0 | 0.0 |
| n48 | 14150800 | OR | WINBERRY CREEK NEAR LOWELL,OR | 0.803 | 1.0 | 1.0 | 1.0 | 0.0 |
| n49 | 09149500 | CO | UNCOMPAHGRE RIVER AT DELTA, CO | 0.763 | 1.0 | 0.997 | 1.0 | 0.0 |
| n50 | 04208000 | OH | Cuyahoga River at Independence OH | 0.566 | 1.0 | 0.877 | 1.0 | 0.0 |
| n51 | 06623800 | WY | ENCAMPMENT RIVER AB HOG PARK CR, N | 0.568 | 1.0 | 0.964 | 1.0 | 0.0 |
| n52 | 03456991 | NC | PIGEON RIVER NEAR CANTON, NC | 0.561 | 1.0 | 1.0 | 1.0 | 0.995 |
| n53 | 02176930 | GA | CHATTOOGA RIVER AT BURRELLS FORD,  | 0.653 | 0.753 | 0.971 | 1.0 | 0.738 |
| n54 | 13317660 | ID | SNAKE RIVER AT McDUFF RAPIDS AT CH | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| n55 | 01480617 | PA | West Branch Brandywine Creek at Mo | 0.813 | 1.0 | 0.992 | 1.0 | 0.0 |
| n56 | 02336728 | GA | UTOY CREEK AT GREAT SOUTHWEST PKWY | 0.978 | 1.0 | 0.985 | 1.0 | 0.978 |
| n57 | 03047000 | PA | Loyalhanna Creek at Loyalhanna Dam | 0.847 | 0.75 | 0.971 | 1.0 | 0.0 |
| n58 | 11467000 | CA | RUSSIAN R A HACIENDA BRIDGE NR GUE | 0.577 | 1.0 | 0.997 | 1.0 | 0.0 |
| n59 | 14211500 | OR | JOHNSON CREEK AT SYCAMORE, OR | 0.999 | 1.0 | 1.0 | 1.0 | 0.0 |
| n60 | 05382257 | WI | STILLWELL CREEK AT YARD ROAD NEAR  | 0.805 | 0.819 | 1.0 | 1.0 | 0.0 |
| n61 | 03106000 | PA | Connoquenessing Creek near Zelieno | 0.655 | 1.0 | 0.99 | 1.0 | 0.0 |
| n62 | 12363000 | MT | Flathead River at Columbia Falls M | 0.995 | 1.0 | 1.0 | 1.0 | 0.0 |
| n63 | 01542500 | PA | WB Susquehanna River at Karthaus,  | 0.665 | 1.0 | 0.996 | 1.0 | 0.0 |
| n64 | 05331833 | WI | NAMEKAGON RIVER AT LEONARDS, WI | 0.854 | 1.0 | 0.985 | 1.0 | 0.0 |
| n65 | 05382255 | WI | STILLWELL CREEK AT SIXTEENTH COURT | 0.781 | 0.819 | 0.967 | 1.0 | 0.0 |
| n66 | 05370000 | WI | EAU GALLE RIVER AT SPRING VALLEY,  | 0.614 | 1.0 | 1.0 | 1.0 | 0.0 |
| n67 | 12213100 | WA | NOOKSACK RIVER AT FERNDALE, WA | 0.867 | 1.0 | 1.0 | 1.0 | 0.994 |
| n68 | 09380000 | AZ | COLORADO RIVER AT LEES FERRY, AZ | 0.993 | 1.0 | 1.0 | 1.0 | 0.0 |
| n69 | 03353200 | IN | EAGLE CREEK AT ZIONSVILLE, IN | 0.672 | 0.999 | 0.997 | 1.0 | 0.0 |
| n70 | 09261000 | UT | GREEN RIVER NEAR JENSEN, UT | 0.968 | 1.0 | 0.997 | 1.0 | 0.0 |
| n71 | 01427207 | PA | DELAWARE RIVER AT LORDVILLE NY | 0.949 | 0.962 | 0.989 | 1.0 | 0.0 |
| n72 | 12212050 | WA | FISHTRAP CREEK AT FRONT STREET AT  | 0.786 | 1.0 | 0.997 | 1.0 | 0.968 |
| n73 | 02334430 | GA | CHATTAHOOCHEE RIVER AT BUFORD DAM, | 0.995 | 1.0 | 1.0 | 1.0 | 0.997 |
| n74 | 08123850 | TX | Colorado Rv abv Silver, TX | 0.788 | 1.0 | 0.951 | 1.0 | 0.987 |
| n75 | 01549700 | PA | Pine Creek bl L Pine Creek near Wa | 0.646 | 1.0 | 1.0 | 1.0 | 0.0 |
| n76 | 03105500 | PA | Beaver River at Wampum, PA | 0.687 | 1.0 | 0.995 | 1.0 | 0.0 |
| n77 | 06934500 | MO | Missouri River at Hermann, MO | 0.859 | 1.0 | 0.982 | 1.0 | 0.0 |
| n78 | 03058000 | WV | WEST FORK R BL STONEWALL JACKSON D | 0.842 | 0.75 | 0.892 | 1.0 | 0.0 |
| n79 | 08048000 | TX | W Fk Trinity Rv at Ft Worth, TX | 0.664 | 1.0 | 0.93 | 1.0 | 0.995 |
| n80 | 02266300 | FL | REEDY CREEK NEAR VINELAND, FL | 0.592 | 1.0 | 0.966 | 1.0 | 0.982 |
| n81 | 14179000 | OR | BREITENBUSH R ABV FRENCH CR NR DET | 0.989 | 1.0 | 0.999 | 1.0 | 0.0 |
| n82 | 01400500 | NJ | Raritan River at Manville NJ | 0.825 | 1.0 | 0.911 | 1.0 | 0.0 |
| n83 | 05517000 | IN | YELLOW RIVER AT KNOX, IN | 0.568 | 1.0 | 0.967 | 1.0 | 0.0 |
| n84 | 05289800 | MN | MINNEHAHA CREEK AT HIAWATHA AVE. I | 0.565 | 0.998 | 0.956 | 0.989 | 0.0 |
| n85 | 06036905 | WY | Firehole River near West Yellowsto | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| n86 | 14091500 | OR | METOLIUS RIVER NEAR GRANDVIEW, OR | 0.999 | 1.0 | 1.0 | 1.0 | 0.0 |
| n87 | 13154500 | ID | SNAKE RIVER AT KING HILL ID | 0.995 | 1.0 | 1.0 | 1.0 | 0.0 |
| n88 | 01104415 | MA | CAMBRIDGE RES., UNNAMED TRIB 2, NR | 0.971 | 1.0 | 0.999 | 1.0 | 0.0 |
| n89 | 07194880 | AR | Osage Creek near Cave Springs, AR | 0.603 | 0.826 | 0.993 | 1.0 | 0.0 |
| n90 | 01073319 | NH | LAMPREY RIVER AT LANGFORD ROAD, AT | 0.595 | 0.834 | 1.0 | 1.0 | 0.0 |
| n91 | 02397000 | GA | COOSA RIVER MAYOS BAR NEAR ROME, G | 0.983 | 1.0 | 0.975 | 1.0 | 0.993 |
| n92 | 02455980 | AL | TURKEY CREEK AT SEWAGE PLANT NEAR  | 0.992 | 1.0 | 0.993 | 1.0 | 0.991 |
| n93 | 09302000 | UT | DUCHESNE RIVER NEAR RANDLETT, UT | 0.985 | 1.0 | 0.995 | 1.0 | 0.0 |
| n94 | 14185000 | OR | SOUTH SANTIAM RIVER BELOW CASCADIA | 0.811 | 1.0 | 1.0 | 1.0 | 0.0 |
| n95 | 10301500 | NV | WALKER RV NR WABUSKA, NV | 0.957 | 1.0 | 1.0 | 1.0 | 0.0 |
| n96 | 02336120 | GA | N.F. PEACHTREE CREEK, BUFORD HWY,  | 0.989 | 1.0 | 0.986 | 1.0 | 0.989 |
| n97 | 255534081324000 | FL | PUMPKIN RIVER NEAR GOODLAND, FL | 0.788 | 0.78 | 0.992 | 0.974 | 0.0 |
| n98 | 03036000 | PA | Mahoning Creek at Mahoning Creek D | 0.778 | 0.75 | 0.977 | 1.0 | 0.0 |
| n99 | 01129500 | NH | CONNECTICUT RIVER AT NORTH STRATFO | 0.858 | 1.0 | 1.0 | 1.0 | 0.0 |
| n100 | 06036805 | WY | Firehole River at Old Faithful, YN | 0.943 | 0.95 | 1.0 | 1.0 | 0.0 |
| n101 | 09163500 | CO | COLORADO RIVER NEAR COLORADO-UTAH  | 0.986 | 1.0 | 1.0 | 1.0 | 0.0 |
| n102 | 10129900 | UT | SILVER CREEK NEAR SILVER CREEK JUN | 0.995 | 1.0 | 1.0 | 1.0 | 0.0 |
| n103 | 14154500 | OR | ROW RIVER ABOVE PITCHER CREEK, NEA | 0.722 | 1.0 | 1.0 | 1.0 | 0.0 |
| n104 | 05406457 | WI | BLACK EARTH CREEK NR BREWERY RD AT | 0.738 | 0.75 | 0.959 | 1.0 | 0.0 |
| n105 | 05458300 | IA | Cedar River at Waverly, IA | 0.567 | 1.0 | 0.803 | 1.0 | 0.0 |
| n106 | 04124200 | MI | MANISTEE RIVER NEAR MESICK, MI | 0.992 | 1.0 | 1.0 | 1.0 | 0.0 |
| n107 | 04067500 | WI | MENOMINEE RIVER NEAR MC ALLISTER,  | 0.634 | 1.0 | 0.969 | 1.0 | 0.0 |
| n108 | 04043244 | MI | EAST BRANCH SALMON TROUT RIVER NEA | 0.987 | 1.0 | 1.0 | 1.0 | 0.0 |
| n109 | 01408029 | NJ | Manasquan River near Allenwood NJ | 0.886 | 1.0 | 0.927 | 1.0 | 0.0 |
| n110 | 03007800 | PA | Allegheny River at Port Allegany,  | 0.664 | 1.0 | 1.0 | 1.0 | 0.0 |
| n111 | 01425000 | NY | WEST BRANCH DELAWARE RIVER AT STIL | 0.983 | 1.0 | 1.0 | 1.0 | 0.0 |
| n112 | 11262900 | CA | MUD SLOUGH NR GUSTINE CA | 0.862 | 1.0 | 0.988 | 1.0 | 0.0 |
| n113 | 14210000 | OR | CLACKAMAS RIVER AT ESTACADA, OR | 0.954 | 1.0 | 0.993 | 1.0 | 0.0 |
| n114 | 03544970 | GA | HIWASSEE RIVER AT RIVERSIDE DR, NR | 0.592 | 0.888 | 0.871 | 1.0 | 0.883 |
| n115 | 04176500 | MI | RIVER RAISIN NEAR MONROE, MI | 0.642 | 1.0 | 1.0 | 1.0 | 0.0 |
| n116 | 01104475 | MA | STONY BROOK RES., UNNAMED TRIB 1,  | 0.997 | 1.0 | 1.0 | 1.0 | 0.0 |
| n117 | 06041000 | MT | Madison River bl Ennis Lake nr McA | 0.989 | 1.0 | 1.0 | 1.0 | 0.0 |
| n118 | 06036940 | WY | Tantalus Creek at Norris Junction, | 0.969 | 1.0 | 0.9 | 1.0 | 0.0 |
| n119 | 01608500 | WV | SOUTH BRANCH POTOMAC RIVER NEAR SP | 0.988 | 1.0 | 0.992 | 1.0 | 0.0 |

## Rejection summary (1345 candidates probed and rejected)

| reason | count |
|---|---|
| no NWIS WTEMP+FLOW | 950 |
| low full-period coverage | 376 |
| low blind-test-period coverage | 19 |

Full per-site detail: `data_usgs/rejected_sites_120v2.csv`.

Meteorology from Daymet single-pixel (TEMP=mean of tmax/tmin, PRCP, DH=solar radiation W/m², RHMEAN from vapour pressure). WDSP not in Daymet — left missing (imputed) and can be added from gridMET.

Combined panel: `data_usgs/panel_usgs.parquet` (657480 rows, 120 sites).
