# Pull public AFL and VFL player-match stats from the AFL/Champion Data API via fitzRoy.
# Both competitions share the same Champion Data player ID, which is what lets us
# follow a player from Box Hill (VFL) to Hawthorn (AFL).
#
# Run from the project folder:   Rscript R/pull_data.R
# Takes roughly 10-20 minutes. Output: data/raw/<comp>_<season>.csv.gz

if (!requireNamespace("fitzRoy", quietly = TRUE)) {
  install.packages("fitzRoy", repos = "https://cloud.r-project.org")
}
library(fitzRoy)
message("fitzRoy version: ", packageVersion("fitzRoy"))

out_dir <- file.path("data", "raw")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

jobs <- rbind(
  data.frame(comp = "AFLM", season = 2015:2026),
  data.frame(comp = "VFL",  season = c(2018, 2019, 2021:2026))  # no VFL season in 2020
)

log <- data.frame()
for (i in seq_len(nrow(jobs))) {
  comp <- jobs$comp[i]; s <- jobs$season[i]
  f <- file.path(out_dir, sprintf("%s_%d.csv.gz", comp, s))
  if (file.exists(f)) { message("skip ", f, " (exists)"); next }
  message(comp, " ", s, " ...")
  df <- tryCatch(fetch_player_stats_afl(season = s, comp = comp),
                 error = function(e) { message("  failed: ", conditionMessage(e)); NULL })
  n <- if (is.null(df)) 0 else nrow(df)
  if (n > 0) {
    df <- as.data.frame(df)
    df <- df[, !vapply(df, is.list, logical(1)), drop = FALSE]  # drop nested columns
    con <- gzfile(f, "w"); write.csv(df, con, row.names = FALSE); close(con)
  }
  message("  rows: ", n)
  log <- rbind(log, data.frame(comp = comp, season = s, rows = n))
}

# AFL list (squad) membership by season, 2015 onwards. This is what tells us which
# VFL players were AFL-listed at the time (including those yet to debut).
lf <- file.path(out_dir, "afl_lists.csv.gz")
if (!file.exists(lf)) {
  message("AFL lists 2015-2026 ...")
  lists <- tryCatch(
    purrr::map_dfr(2015:2026, function(s) {
      d <- fetch_player_details(season = s, current = TRUE, comp = "AFLM", source = "AFL")
      if (is.null(d) || nrow(d) == 0) return(NULL)
      d <- as.data.frame(d); d <- d[, !vapply(d, is.list, logical(1)), drop = FALSE]
      d$list_season <- s; d
    }),
    error = function(e) { message("  lists failed: ", conditionMessage(e)); NULL })
  if (!is.null(lists)) { con <- gzfile(lf, "w"); write.csv(lists, con, row.names = FALSE); close(con)
    message("  list rows: ", nrow(lists)) }
}

write.csv(log, file.path(out_dir, "pull_log.csv"), row.names = FALSE)
print(log)
message("Done. Files in ", normalizePath(out_dir))
