require(raster);require(readxl);require(dismo);require(dplyr);require(plyr)

dat <- read_excel("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/occurrence.xlsx",sheet = "US_F")
 coordinates(dat) <- ~decimalLongitude+decimalLatitude

temp <- raster("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/wc2.0_bio_2.5m_01.tif")
prec <- raster("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/wc2.0_bio_2.5m_12.tif")

bios <- stack(temp,prec)

data <- raster::extract(bios,dat)
colnames(data) <- c("bio_1","bio_12")

dups <- duplicated(data[, c("bio_1","bio_12")])
# remove duplicates
data2 <- data[!dups, ]
data2 <- as.data.frame(data2)
plot(data2)

set.seed(1111)
data3 <- sample_n(data2, 100)
data3_a <- data3
data3_a$id <- 1:nrow(data3_a)
# plot(data3)
write.csv(data3,"E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int.csv",quote = F,na = "",row.names = F)

data3_a_T <- sample_n(data3_a, 50)
data3_a_Test <- anti_join(data3_a,data3_a_T,by="id")
data3_a_T$id <- NULL
data3_a_Test$id <- NULL
write.csv(data3_a_T,"E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_training.csv",quote = F,na = "",row.names = F)
write.csv(data3_a_Test,"E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_testing.csv",quote = F,na = "",row.names = F)

