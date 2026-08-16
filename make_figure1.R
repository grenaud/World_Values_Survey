library(magick)

BG <- "white"
LABEL_SIZE <- 80
LABEL_OFFSET <- "+20+20"
TARGET_H <- 1200  # adjust if you want larger/smaller outputs

label_panel <- function(img, letter,
                        frac = 0.06,     # ~6% of panel height
                        min_size = 40,
                        max_size = 110) {
  h <- image_info(img)$height
  sz <- as.integer(round(h * frac))
  sz <- max(min_size, min(max_size, sz))

  image_annotate(
    img, letter,
    gravity = "northwest",
    location = LABEL_OFFSET,  # keep your "+20+20"
    size = sz,
    color = "black",
    strokecolor = "white",
    boxcolor = NULL
  )
}

# Safe read: decode at reduced size if huge (uses identify)
read_png_scaled <- function(path, target_h = TARGET_H) {
  dims <- system2("identify", args = c("-format", "%w,%h", shQuote(path)), stdout = TRUE)
  parts <- strsplit(trimws(dims[1]), ",")[[1]]
  w <- as.integer(parts[1]); h <- as.integer(parts[2])

  if (!is.null(target_h) && h > target_h) {
    scale <- target_h / h
    new_w <- max(1L, as.integer(round(w * scale)))
    new_h <- max(1L, as.integer(round(h * scale)))
    image_read(sprintf("%s[%dx%d]", path, new_w, new_h))
  } else {
    image_read(path)
  }
}

# ---------- Figure 1 (special layout rules) ----------
A <- read_png_scaled("WVS_v6/language_families_map.png")
B <- read_png_scaled("WVS_v6/language_families_pie_chart.png")
C <- read_png_scaled("WVS_v6/questions_categories_pie_chart.png")

# Scale B and C to same height (no cropping)
target_bc_h <- min(image_info(B)$height, image_info(C)$height)
B <- image_scale(B, paste0("x", target_bc_h))
C <- image_scale(C, paste0("x", target_bc_h))

# Add labels (consistent offsets)
A <- label_panel(A, "A)")
B <- label_panel(B, "B)")
C <- label_panel(C, "C)")

# Bottom row
bottom <- image_append(c(B, C), stack = FALSE)

# Make final figure as wide as A (top panel):
A_w <- image_info(A)$width

# If bottom is wider than A, scale bottom down to A_w (keeps aspect, no crop)
if (image_info(bottom)$width > A_w) {
  bottom <- image_scale(bottom, paste0(A_w, "x"))
}

# If bottom is narrower than A, pad it to A_w (centered)
bottom <- image_extent(
  bottom,
  geometry = paste0(A_w, "x", image_info(bottom)$height),
  gravity = "center",
  color = BG
)

# Also pad A to A_w (no-op, but keeps logic symmetric)
A <- image_extent(
  A,
  geometry = paste0(A_w, "x", image_info(A)$height),
  gravity = "center",
  color = BG
)

# Stack vertically
fig1 <- image_append(c(A, bottom), stack = TRUE)

image_write(fig1, path = "Figure_1.png", format = "png")
cat("[OK] rewrote Figure_1.png\n")
