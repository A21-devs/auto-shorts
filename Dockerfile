# clipwright with ffmpeg and every extra except gpu.
#   docker build -t clipwright .
#   docker run --rm -v "$PWD:/work" clipwright "/work/my stream.mp4" -n 6
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[transcribe,link,speaker,mcp]"

WORKDIR /work
ENTRYPOINT ["clipwright"]
CMD ["--help"]
