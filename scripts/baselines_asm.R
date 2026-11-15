# E4, R half: antitonic score matching (package asm) on the samples saved by baselines_mc.py.
# Usage: Rscript baselines_asm.R <data_dir> <cores> [R library holding asm]
# Install once: install.packages("asm")   (version used: 0.2.4)
# Reads data_<law>_<n>_<R>.bin (float32, shape R x n x 5: four regressors, response) and writes
# asm_<law>_<n>_<R>.txt with the slope sum of squares per replication for
#   asm_asym: asm.fit(X, y, symmetric = FALSE)   asm_sym: asm.fit(X, y)  (package default)
# Both use the package's default LAD pilot. A failed fit is recorded as NA.
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
cores <- as.integer(args[2])
if (length(args) > 2) library(asm, lib.loc = args[3]) else library(asm)

sse <- function(X, y, sym) {
  tryCatch({
    b <- asm.fit(X, y, symmetric = sym)$betahat
    sum(b[2:5]^2)
  }, error = function(e) NA_real_)
}

run <- function(path) {
  id <- sub("^data_", "", sub("\\.bin$", "", basename(path)))
  v <- as.integer(strsplit(id, "_")[[1]])
  n <- v[2]; R <- v[3]
  con <- file(path, "rb")
  a <- array(readBin(con, "numeric", n = R * n * 5, size = 4), dim = c(5, n, R))
  close(con)
  out <- matrix(NA_real_, R, 2, dimnames = list(NULL, c("asm_asym", "asm_sym")))
  for (r in seq_len(R)) {
    d <- t(a[, , r])
    X <- cbind(1, d[, 1:4])
    out[r, 1] <- sse(X, d[, 5], FALSE)
    out[r, 2] <- sse(X, d[, 5], TRUE)
  }
  write.table(out, file.path(dir, paste0("asm_", id, ".txt")), row.names = FALSE, quote = FALSE)
  id
}

files <- Sys.glob(file.path(dir, "data_*.bin"))
print(unlist(parallel::mclapply(files, run, mc.cores = cores)))
