#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(magick))

BG <- "white"
TARGET_H <- 1200          # reduce to 800 if needed
DENSITY <- "72x72"        # critical: makes point sizes comparable across panels

BAND_H <- 90              # fixed band height in px (same for all)
LABEL_PT <- 48            # point size (consistent when density is fixed)
XOFF <- 30                # px from left
YOFF <- 18                # px from top
FONT <- "DejaVu-Sans-Bold"

# Safe read: query size with identify, then decode downscaled if needed
# and force the same density at read time.
read_png_scaled <- function(path, target_h = TARGET_H, density = DENSITY) {
  if (!file.exists(path)) {
    stop(sprintf("File not found: %s\nWorking directory: %s", path, getwd()), call. = FALSE)
  }

  dims <- system2("identify", args = c("-format", "%w,%h", shQuote(path)), stdout = TRUE)
  parts <- strsplit(trimws(dims[1]), ",")[[1]]
  w <- as.integer(parts[1]); h <- as.integer(parts[2])

  if (!is.null(target_h) && h > target_h) {
    scale <- target_h / h
    new_w <- max(1L, as.integer(round(w * scale)))
    new_h <- max(1L, as.integer(round(h * scale)))
    image_read(sprintf("%s[%dx%d]", path, new_w, new_h), density = density)
  } else {
    image_read(path, density = density)
  }
}

# Scale images to same height (no cropping)
scale_to_common_height <- function(imgs) {
  hs <- sapply(imgs, function(x) image_info(x)$height)
  target_h <- min(hs)
  lapply(imgs, function(x) image_scale(x, paste0("x", target_h)))
}

# Create a label band at the same density, then annotate inside it
make_band <- function(width, height, density = DENSITY) {
  # "xc:white" is an ImageMagick canvas. We then extent to desired size.
  band <- image_read(sprintf("xc:%s", BG), density = density)
  image_extent(band, geometry = paste0(width, "x", height), gravity = "northwest", color = BG)
}

add_label_band <- function(img, label) {
  info <- image_info(img)
  band <- make_band(info$width, BAND_H)

  band <- image_annotate(
    band, label,
    gravity = "northwest",
    location = paste0("+", XOFF, "+", YOFF),
    size = LABEL_PT,
    font = FONT,
    color = "black",
    strokecolor = "white",
    boxcolor = NULL
  )

  image_append(c(band, img), stack = TRUE)
}

# ---------- Figure 2 ----------
pA <- "WVS_v6/filtered_clustermaplang_imp.png"
pB <- "WVS_v6/filtered_pca_pc1_pc2_imp.png"
pC <- "WVS_v6/filtered_biplotPCA_circ_Religion.png"
pD <- "WVS_v6/filtered_biplotPCA_circ_Politics.png"

A0 <- read_png_scaled(pA)
B0 <- read_png_scaled(pB)
C0 <- read_png_scaled(pC)
D0 <- read_png_scaled(pD)

# Make ALL four panels the same height BEFORE labeling
imgs <- scale_to_common_height(list(A0, B0, C0, D0))
A <- add_label_band(imgs[[1]], "A)")
B <- add_label_band(imgs[[2]], "B)")
C <- add_label_band(imgs[[3]], "C)")
D <- add_label_band(imgs[[4]], "D)")

top <- image_append(c(A, B), stack = FALSE)
bottom <- image_append(c(C, D), stack = FALSE)

# Final width driven by top row; bottom scaled/padded to match (no cropping)
TOPW <- image_info(top)$width
if (image_info(bottom)$width > TOPW) {
  bottom <- image_scale(bottom, paste0(TOPW, "x"))
}

top <- image_extent(top, geometry = paste0(TOPW, "x", image_info(top)$height),
                    gravity = "center", color = BG)
bottom <- image_extent(bottom, geometry = paste0(TOPW, "x", image_info(bottom)$height),
                       gravity = "center", color = BG)

fig2 <- image_append(c(top, bottom), stack = TRUE)
image_write(fig2, "Figure_2.png", format = "png")
cat("[OK] wrote Figure_2.png\n")
