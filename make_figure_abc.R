#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(magick))

# ---------------- user knobs ----------------
BG <- "white"
LABEL_SIZE <- 80
LABEL_OFFSET <- "+20+20"

# Key fix: keep this modest to avoid IM cache exhaustion.
# Typical good values: 800–1600 depending on desired output resolution.
TARGET_H <- 1200

read_png_scaled <- function(path, target_h = TARGET_H) {
  if (!file.exists(path)) {
    stop(sprintf("File not found: %s\nWorking directory: %s", path, getwd()), call. = FALSE)
  }

  # Get width/height without loading pixels
  dims <- system2(
    "identify",
    args = c("-format", "%w,%h", shQuote(path)),
    stdout = TRUE,
    stderr = TRUE
  )

  if (length(dims) == 0 || grepl("^identify:|error", tolower(dims[1]))) {
    stop(sprintf("identify failed for: %s\nOutput: %s", path, paste(dims, collapse=" ")), call. = FALSE)
  }

  parts <- strsplit(trimws(dims[1]), ",")[[1]]
  w <- as.integer(parts[1])
  h <- as.integer(parts[2])

  # Decode already resized if needed (prevents cache exhaustion)
  if (!is.null(target_h) && h > target_h) {
    scale <- target_h / h
    new_w <- max(1L, as.integer(round(w * scale)))
    new_h <- max(1L, as.integer(round(h * scale)))
    image_read(sprintf("%s[%dx%d]", path, new_w, new_h))
  } else {
    image_read(path)
  }
}

label_panel <- function(img, letter, size = LABEL_SIZE) {
  image_annotate(
    img, letter,
    gravity = "northwest",
    location = LABEL_OFFSET,
    size = size,
    color = "black",
    strokecolor = "white",
    boxcolor = NULL
  )
}

scale_to_common_height <- function(imgs) {
  hs <- sapply(imgs, function(x) image_info(x)$height)
  target_h <- min(hs)
  lapply(imgs, function(x) image_scale(x, paste0("x", target_h)))
}

pad_to <- function(img, w, h, bg = BG) {
  image_extent(img, geometry = paste0(w, "x", h), gravity = "center", color = bg)
}

make_row <- function(paths, labels = NULL, label_size = LABEL_SIZE) {
  imgs <- lapply(paths, read_png_scaled)
  imgs <- scale_to_common_height(imgs)

  if (!is.null(labels)) {
    if (length(labels) != length(imgs)) {
      stop("labels length must match number of images in row", call. = FALSE)
    }
    imgs <- Map(function(im, lab) label_panel(im, lab, size = label_size), imgs, labels)
  }

  image_append(do.call(c, imgs), stack = FALSE)  # horizontal
}

make_grid <- function(row_specs, out, bg = BG) {
  rows <- lapply(row_specs, function(rs) {
    make_row(
      paths = rs$paths,
      labels = rs$labels,
      label_size = if (!is.null(rs$label_size)) rs$label_size else LABEL_SIZE
    )
  })

  widths  <- sapply(rows, function(r) image_info(r)$width)
  heights <- sapply(rows, function(r) image_info(r)$height)
  max_w <- max(widths)

  rows_padded <- Map(function(r, h) pad_to(r, max_w, h, bg), rows, heights)
  fig <- image_append(do.call(c, rows_padded), stack = TRUE)  # vertical

  image_write(fig, path = out, format = "png")
  message("[OK] wrote ", out)
}

# ---------------- optional: check files exist ----------------
required_files <- c(
  "WVS_v6/language_families_map.png",
  "WVS_v6/language_families_pie_chart.png",
  "WVS_v6/questions_categories_pie_chart.png",
  "WVS_v6/filtered_clustermaplang_imp.png",
  "WVS_v6/filtered_pca_pc1_pc2_imp.png",
  "WVS_v6/filtered_biplotPCA_circ_Religion.png",
  "WVS_v6/filtered_biplotPCA_circ_Politics.png",
  "WVS_v6/filtered_nmf_c2_imp.png",
  "WVS_v6/filtered_nmf_c3_imp.png",
  "WVS_v6/filtered_nmf_c3_triangle.png",
  "WVS_v6/CANKAZ_filter_pca_pc1_pc2_imp.png",
  "WVS_v6/CANKAZ_filter_clustermaplang_imp.png",
  "WVS_v6/CANKAZ_filter_distances_sorted_KZK.png",
  "WVS_v6/CANKAZ_filter_distances_sorted_KZR.png",
  "WVS_v6/CANKAZ_filter_distances_sorted_CDE.png",
  "WVS_v6/CANKAZ_filter_distances_sorted_CDF.png",
  "WVS_v6/USAPolitics_filter_pca_pc1_pc2_imp.png",
  "WVS_v6/USAPolitics_filter_distances_sorted_USD.png",
  "WVS_v6/USAPolitics_filter_distances_sorted_USR.png",
  "WVS_v6/16-29_CADUSA_distances_sorted_CDE.png",
  "WVS_v6/50andOver_CADUSA_distances_sorted_CDE.png"
)

missing <- required_files[!file.exists(required_files)]
if (length(missing) > 0) {
  stop(
    paste0("Missing files:\n", paste0("  - ", missing, collapse = "\n"),
           "\nWorking directory: ", getwd()),
    call. = FALSE
  )
}

# ---------------- build figures ----------------

# Figure 1: top A; bottom B|C
make_grid(
  row_specs = list(
    list(paths = c("WVS_v6/language_families_map.png"), labels = c("A)")),
    list(paths = c("WVS_v6/language_families_pie_chart.png",
                   "WVS_v6/questions_categories_pie_chart.png"),
         labels = c("B)", "C)"))
  ),
  out = "Figure_1.png"
)

# Figure 2: top A|B; bottom C|D
make_grid(
  row_specs = list(
    list(paths = c("WVS_v6/filtered_clustermaplang_imp.png",
                   "WVS_v6/filtered_pca_pc1_pc2_imp.png"),
         labels = c("A)", "B)")),
    list(paths = c("WVS_v6/filtered_biplotPCA_circ_Religion.png",
                   "WVS_v6/filtered_biplotPCA_circ_Politics.png"),
         labels = c("C)", "D)"))
  ),
  out = "Figure_2.png"
)

# Figure 3: top A|B; bottom C
make_grid(
  row_specs = list(
    list(paths = c("WVS_v6/filtered_nmf_c2_imp.png",
                   "WVS_v6/filtered_nmf_c3_imp.png"),
         labels = c("A)", "B)")),
    list(paths = c("WVS_v6/filtered_nmf_c3_triangle.png"),
         labels = c("C)"))
  ),
  out = "Figure_3.png"
)

# Figure 4: top A|B
make_grid(
  row_specs = list(
    list(paths = c("WVS_v6/CANKAZ_filter_pca_pc1_pc2_imp.png",
                   "WVS_v6/CANKAZ_filter_clustermaplang_imp.png"),
         labels = c("A)", "B)"))
  ),
  out = "Figure_4.png"
)

# Figure 5: top A|B; bottom C|D
make_grid(
  row_specs = list(
    list(paths = c("WVS_v6/CANKAZ_filter_distances_sorted_KZK.png",
                   "WVS_v6/CANKAZ_filter_distances_sorted_KZR.png"),
         labels = c("A)", "B)")),
    list(paths = c("WVS_v6/CANKAZ_filter_distances_sorted_CDE.png",
                   "WVS_v6/CANKAZ_filter_distances_sorted_CDF.png"),
         labels = c("C)", "D)"))
  ),
  out = "Figure_5.png"
)

# Figure 6: row1 A; row2 B|C; row3 D|F
make_grid(
  row_specs = list(
    list(paths = c("WVS_v6/USAPolitics_filter_pca_pc1_pc2_imp.png"),
         labels = c("A)")),
    list(paths = c("WVS_v6/USAPolitics_filter_distances_sorted_USD.png",
                   "WVS_v6/USAPolitics_filter_distances_sorted_USR.png"),
         labels = c("B)", "C)")),
    list(paths = c("WVS_v6/16-29_CADUSA_distances_sorted_CDE.png",
                   "WVS_v6/50andOver_CADUSA_distances_sorted_CDE.png"),
         labels = c("D)", "F)"))
  ),
  out = "Figure_6.png"
)


