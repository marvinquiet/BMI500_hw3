library(Matrix)
# load count matrix and cell-type annotation
count_data <- readRDS("FC_data_LRcell.RDS")
annotation <- readRDS("FC_cells_annot.RDS")

# sample 4000 cells from 71,639 cells
N <- 4000

annot_tab <- table(annotation)
cell_list <- c()
for (cluster in names(annot_tab)) {
    num <- round(annot_tab[cluster]/71639*N)
    set.seed(2020)
    cells <- names(sample(annotation[grepl(cluster, annotation)], num))
    cell_list <- c(cell_list, cells)
}

sampled_annot <- annotation[cell_list]
sampled_count <- count_data[, cell_list]

write.table(as.matrix(sampled_count), file="sampled_FC_data.csv", sep=",", quote=F)
write.table(as.data.frame(sampled_annot), file="sampled_FC_annot.csv", sep=",", quote=F)
