library("usdarnass")

nass_set_key("2D21ED3F-190F-3DE8-8955-F18A8E68B61B")
readRenviron("~/.Renviron")
Sys.getenv("NASS_KEY")

nass_param('state_name')

crops1 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE', 
                    state_name = "ALABAMA" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops1, "crops1.csv")

crops2 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE', 
                    state_name = "ALASKA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops2, "crops2.csv")

crops3 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE', 
                    state_name = "ARIZONA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops3, "crops3.csv")

crops4 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                    state_name = "ARKANSAS",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops4, "crops4.csv")

crops5 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                    state_name = "FLORIDA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops5, "crops5.csv")

crops6 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                    state_name = "DELAWARE",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops6, "crops6.csv")

crops7 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                    state_name = "WEST VIRGINIA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops7, "crops7.csv")

crops8 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                    state_name = "WASHINGTON",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops8, "crops8.csv")

crops9 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                    state_name =  "WYOMING",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops9, "crops9.csv")

crops10 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "WISCONSIN",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops10, "crops10.csv")

crops11 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "VERMONT",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops11, "crops11.csv")

crops12 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "VIRGINIA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops12, "crops12.csv")

crops13 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "TENNESSEE" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops13, "crops13.csv")

crops14 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "TEXAS" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops14, "crops14.csv")

crops15 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "UTAH",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops15, "crops15.csv")

crops16 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "SOUTH CAROLINA" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops16, "crops16.csv")

crops17 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "RHODE ISLAND" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops17, "crops17.csv")

crops18 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "SOUTH DAKOTA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops18, "crops18.csv")

crops19 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "PENNSYLVANIA"  ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops19, "crops19.csv")

crops20 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "OHIO",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops20, "crops20.csv")

crops21 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "NORTH DAKOTA" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops21, "crops21.csv")

crops22 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "OKLAHOMA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops22, "crops22.csv")

crops23 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "OREGON", year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops23, "crops23.csv")

crops24 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "NEW HAMPSHIRE",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops24, "crops24.csv")

crops25 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "NEW JERSEY" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops25, "crops25.csv")

crops26 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "NEW MEXICO",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops26, "crops26.csv")

crops27 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "NEW YORK"  ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops27, "crops27.csv")

crops28 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "NORTH CAROLINA" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops28, "crops28.csv")

crops29 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "MONTANA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops29, "crops29.csv")

crops30 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "MINNESOTA" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops30, "crops30.csv")

crops31 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "MISSISSIPPI",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops31, "crops31.csv")

crops32 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "NEBRASKA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops32, "crops32.csv")

crops33 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "NEVADA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops33, "crops33.csv")

crops34 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "MICHIGAN" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops34, "crops34.csv")

crops35 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "MISSOURI",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops35, "crops35.csv")

crops36 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "MAINE" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops36, "crops36.csv")

crops37 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "MARYLAND",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops37, "crops37.csv")

crops38 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "IOWA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops38, "crops38.csv")

crops39 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "MASSACHUSETTS",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops39, "crops39.csv")

crops40 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "MICHIGAN",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops40, "crops40.csv")

crops41 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "KANSAS",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops41, "crops41.csv")

crops42 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "INDIANA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops42, "crops42.csv")

crops43 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "HAWAII" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops43, "crops43.csv")

crops44 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "COLORADO",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops44, "crops44.csv")

crops45 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "GEORGIA" ,year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops45, "crops45.csv")

crops46 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "IDAHO",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops46, "crops46.csv")

crops47 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "KENTUCKY",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops47, "crops47.csv")

crops48 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "LOUISIANA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops48, "crops48.csv")

crops49 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "CONNECTICUT",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops49, "crops49.csv")

crops50 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "ILLINOIS",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops50, "crops50.csv")

crops51 <- nass_data(sector_desc = 'CROPS',agg_level_desc = 'STATE',  
                     state_name = "CALIFORNIA",year = "2007",  freq_desc = 'ANNUAL',reference_period_desc = 'YEAR')
write.csv(crops51, "crops51.csv")
