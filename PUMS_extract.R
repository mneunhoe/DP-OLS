data("PUMS5extract10000")

dat <- PUMS5extract10000

dat <- dat[,4:11]
str(dat)
means <- colMeans(dat)
sds <- apply(dat, 2, sd)
dat_z <- sweep(sweep(dat[,2:4], 2, means[2:4]), 2, sds[2:4], "/")
dat[, 2:4] <- dat_z
colnames(dat)
summary(dat)
saveRDS(list(dat_z, means, sds), "PUMS5extract_z.RDS")
write.csv(dat, "PUMS5extract_z.csv", row.names = F)
