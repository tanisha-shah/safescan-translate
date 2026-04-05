FROM libretranslate/libretranslate:latest
ENV LT_LOAD_ONLY=en,hi
ENV LT_API_KEYS=false
EXPOSE 5000
