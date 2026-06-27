# USGS large-sample acquisition (40 stations)

_Window 2006-01-01…2020-12-31. Probed 364 candidates in 4237s. Schema matches the original study._

| site_id | USGS | state | name | WTEMP cov | FLOW cov | WLEVEL cov |
|---|---|---|---|---|---|---|
| n00 | 04092750 | IN | INDIANA HARBOR CANAL AT EAST CHICA | 0.626 | 0.996 | 0.0 |
| n01 | 04027000 | WI | BAD RIVER NEAR ODANAH, WI | 0.613 | 1.0 | 0.0 |
| n02 | 04213500 | NY | CATTARAUGUS CREEK AT GOWANDA NY | 0.637 | 1.0 | 0.0 |
| n03 | 11261500 | CA | SAN JOAQUIN R A FREMONT FORD BRIDG | 0.858 | 1.0 | 0.0 |
| n04 | 06486000 | NE | Missouri River at Sioux City, IA | 0.609 | 1.0 | 0.0 |
| n05 | 05382284 | WI | SILVER CREEK AT STATE HIGHWAY 21 N | 0.681 | 0.761 | 0.0 |
| n06 | 09234500 | UT | GREEN RIVER NEAR GREENDALE, UT | 0.982 | 1.0 | 0.0 |
| n07 | 05382267 | WI | LA CROSSE RIVER @ CNTY TRUNK HIGHW | 0.751 | 0.819 | 0.0 |
| n08 | 10133800 | UT | EAST CANYON CREEK NEAR JEREMY RANC | 0.995 | 1.0 | 0.0 |
| n09 | 11507500 | OR | LINK RIVER AT KLAMATH FALLS, OR | 0.985 | 1.0 | 0.0 |
| n10 | 03353611 | IN | WHITE R. AT STOUT GEN. STN. AT IND | 0.898 | 1.0 | 0.0 |
| n11 | 09163500 | CO | COLORADO RIVER NEAR COLORADO-UTAH  | 0.986 | 1.0 | 0.0 |
| n12 | 12213100 | WA | NOOKSACK RIVER AT FERNDALE, WA | 0.867 | 1.0 | 0.994 |
| n13 | 06768000 | NE | Platte River near Overton, Nebr. | 0.592 | 1.0 | 0.942 |
| n14 | 14159500 | OR | SOUTH FORK MCKENZIE RIVER NEAR RAI | 0.999 | 1.0 | 0.0 |
| n15 | 06066500 | MT | Missouri River bl Holter Dam nr Wo | 0.998 | 1.0 | 0.0 |
| n16 | 10129900 | UT | SILVER CREEK NEAR SILVER CREEK JUN | 0.995 | 1.0 | 0.0 |
| n17 | 03039000 | PA | Crooked Creek at Crooked Creek Dam | 0.776 | 0.75 | 0.0 |
| n18 | 14178000 | OR | NO SANTIAM R BLW BOULDER CRK, NR D | 0.999 | 1.0 | 0.0 |
| n19 | 05406457 | WI | BLACK EARTH CREEK NR BREWERY RD AT | 0.738 | 0.75 | 0.0 |
| n20 | 09196500 | WY | PINE CREEK ABOVE FREMONT LAKE, WY | 0.558 | 1.0 | 0.0 |
| n21 | 12115000 | WA | CEDAR RIVER NEAR CEDAR FALLS, WA | 0.922 | 1.0 | 0.375 |
| n22 | 03353200 | IN | EAGLE CREEK AT ZIONSVILLE, IN | 0.672 | 0.999 | 0.0 |
| n23 | 14211499 | OR | KELLEY CREEK AT SE 159TH DRIVE AT  | 0.966 | 1.0 | 0.0 |
| n24 | 02077303 | NC | HYCO R BL ABAY D NR MCGEHEES MILL, | 0.855 | 0.988 | 0.96 |
| n25 | 04043244 | MI | EAST BRANCH SALMON TROUT RIVER NEA | 0.987 | 1.0 | 0.0 |
| n26 | 02011400 | VA | JACKSON RIVER NEAR BACOVA, VA | 0.939 | 1.0 | 0.0 |
| n27 | 13016450 | WY | FISH CREEK AT WILSON, WY | 0.683 | 1.0 | 0.0 |
| n28 | 13068500 | ID | BLACKFOOT RIVER NR BLACKFOOT ID | 0.645 | 0.937 | 0.0 |
| n29 | 04125550 | MI | MANISTEE RIVER NEAR WELLSTON, MI | 0.983 | 1.0 | 0.0 |
| n30 | 12340000 | MT | Blackfoot River near Bonner MT | 0.833 | 1.0 | 0.0 |
| n31 | 14320934 | OR | LITTLE WOLF CREEK NEAR TYEE, OR | 0.866 | 0.889 | 0.0 |
| n32 | 14151000 | OR | FALL CREEK BLW WINBERRY CREEK, NEA | 0.987 | 1.0 | 0.0 |
| n33 | 12205000 | WA | NF NOOKSACK RIVER BL CASCADE CREEK | 0.858 | 1.0 | 0.997 |
| n34 | 14197900 | OR | WILLAMETTE RIVER AT NEWBERG, OR | 0.826 | 1.0 | 0.0 |
| n35 | 06036905 | WY | Firehole River near West Yellowsto | 1.0 | 1.0 | 0.0 |
| n36 | 13056500 | ID | HENRYS FORK NR REXBURG ID | 0.726 | 1.0 | 0.0 |
| n37 | 03054500 | WV | TYGART VALLEY RIVER AT PHILIPPI, W | 0.595 | 1.0 | 0.0 |
| n38 | 12398600 | WA | PEND OREILLE RIVER AT INTERNATIONA | 0.937 | 1.0 | 0.0 |
| n39 | 14316495 | OR | BOULDER CREEK NEAR TOKETEE FALLS,  | 0.694 | 1.0 | 0.0 |

Meteorology from Daymet single-pixel (TEMP=mean of tmax/tmin, PRCP, DH=solar radiation W/m², RHMEAN from vapour pressure). WDSP not in Daymet — left missing (imputed) and can be added from gridMET.

Combined panel: `data_usgs/panel_usgs.parquet` (219160 rows, 40 sites).