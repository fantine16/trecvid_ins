FILE(REMOVE_RECURSE
  "CMakeFiles/perf"
  "dummy_perf"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/perf.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
