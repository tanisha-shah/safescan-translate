FROM libretranslate/libretranslate:latest
ENV LT_LOAD_ONLY=en,hi,gu,ta,bn,te
ENV LT_API_KEYS=false
EXPOSE 5000
