context("teardown")

# remove temporary 'rgf' files
if (dir.exists(default_dir)) unlink(default_dir, recursive = TRUE, force = TRUE)
