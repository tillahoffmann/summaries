# This script converts from RDA to CSV.
args = commandArgs(trailingOnly=TRUE)
if (length(args) != 2) {
    stop("must supply exactly two arguments", call.=FALSE)
}
load(args[1])
write.csv(coaloracle, file=args[2], row.names=FALSE)
