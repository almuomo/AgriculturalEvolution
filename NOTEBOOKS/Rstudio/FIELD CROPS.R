library("usdarnass")

nass_set_key("2D21ED3F-190F-3DE8-8955-F18A8E68B61B")
readRenviron("~/.Renviron")
Sys.getenv("NASS_KEY")

#Ayuda a conocer las variables no númérica de cada parametro
nass_param('source_desc')

# conocer los valores de la petición realizada a la API (max 50.000 valores)
nass_count(sector_desc = 'CROPS',
           agg_level_desc = 'STATE',
           
           year = '1950',
           freq_desc = 'ANNUAL',
           reference_period_desc = 'YEAR')



crops_1950 <- nass_data(sector_desc = 'CROPS',
                   agg_level_desc = 'STATE',
                   
                   year = "1950",
                   freq_desc = 'ANNUAL',
                   reference_period_desc = 'YEAR')
write.csv(crops_1950, "1950.csv")


crops_1951 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1951",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1951, "1951.csv")


crops_1952 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1952",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1952, "1952.csv")


crops_1953 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1953",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1953, "1953.csv")


crops_1954 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1954",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1954, "1954.csv")


crops_1955 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1955",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1955, "1955.csv")


crops_1956 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1956",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1956, "1956.csv")


crops_1957 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1957",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1957, "1957.csv")


crops_1958 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1958",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1958, "1958.csv")


crops_1959 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1959",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1959, "1959.csv")

crops_1960 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1960",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1960, "1960.csv")


crops_1961 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1961",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1961, "1961.csv")


crops_1962 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1962",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1962, "1962.csv")


crops_1963 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1963",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1963, "1963.csv")


crops_1964 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1964",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1964, "1964.csv")


crops_1965 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1965",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1965, "1965.csv")


crops_1966 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1966",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1966, "1966.csv")


crops_1967 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1967",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1967, "1967.csv")


crops_1968 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1968",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1968, "1968.csv")


crops_1969 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1969",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1969, "1969.csv")

crops_1970 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1970",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1970, "1970.csv")


crops_1971 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1971",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1971, "1971.csv")


crops_1972 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1972",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1972, "1972.csv")


crops_1973 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1973",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1973, "1973.csv")


crops_1974 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1974",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1974, "1974.csv")


crops_1975 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1975",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1975, "1975.csv")


crops_1976 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1976",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1976, "1976.csv")


crops_1977 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1977",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1977, "1977.csv")


crops_1978 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1978",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1978, "1978.csv")


crops_1979 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1979",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1979, "1979.csv")


crops_1980 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1980",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1980, "1980.csv")


crops_1981 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1981",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1981, "1981.csv")



crops_1982 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1982",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1982, "1982.csv")



crops_1983 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1983",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1983, "1983.csv")



crops_1984 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1984",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1984, "1984.csv")



crops_1985 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1985",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1985, "1985.csv")



crops_1986 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1986",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1986, "1986.csv")



crops_1987 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1987",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1987, "1987.csv")



crops_1988 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1988",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1988, "1988.csv")



crops_1989 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1989",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1989, "1989.csv")



crops_1990 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1990",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1990, "1990.csv")



crops_1991 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1991",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1991, "1991.csv")



crops_1992 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1992",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1992, "1992.csv")



crops_1993 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1993",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1993, "1993.csv")



crops_1994 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1994",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1994, "1994.csv")



crops_1995 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1995",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1995, "1995.csv")



crops_1996 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1996",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1996, "1996.csv")



crops_1997 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1997",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1997, "1997.csv")



crops_1998 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1998",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1998, "1998.csv")



crops_1999 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "1999",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_1999, "1999.csv")



crops_2000 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2000",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2000, "2000.csv")


crops_2001 <- nass_data(
                          sector_desc = 'CROPS',
                          agg_level_desc = 'STATE',
                          
                          year = "2001",
                          freq_desc = 'ANNUAL',
                          reference_period_desc = 'YEAR')
write.csv(crops_2001, "2001.csv")


#2002


crops_2003 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2003",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2003, "2003.csv")




crops_2004 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2004",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2004, "2004.csv")




crops_2005 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2005",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2005, "2005.csv")




crops_2006 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2006",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2006, "2006.csv")


##2007



crops_2008 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2008",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2008, "2008.csv")


##2009


crops_2010 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2010",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2010, "2010.csv")



crops_2011 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2011",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2011, "2011.csv")


##2012

crops_2013 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       year = "2013",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2013, "2013.csv")

##2014

crops_2015 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2015",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2015, "2015.csv")



crops_2016 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2016",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2016, "2016.csv")

##2017


crops_2018 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2018",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2018, "2018.csv")


##2019

crops_2020 <- nass_data(sector_desc = 'CROPS',
                       agg_level_desc = 'STATE',
                       
                       year = "2020",
                       freq_desc = 'ANNUAL',
                       reference_period_desc = 'YEAR')
write.csv(crops_2020, "2020.csv")

crops_2021 <- nass_data(sector_desc = 'CROPS',
                        agg_level_desc = 'STATE',
                        
                        year = "2021",
                        freq_desc = 'ANNUAL',
                        reference_period_desc = 'YEAR')
write.csv(crops_2021, "2021.csv")

