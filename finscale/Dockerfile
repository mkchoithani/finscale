# Build: docker buildx build --platform linux/arm64 -t <REGISTRY>/finscale:<TAG> .
# TODO: confirm NGC tag with organizers (needs arm64 + Blackwell/GB10 support)
FROM nvcr.io/nvidia/pytorch:25.09-py3
WORKDIR /app
RUN pip install --no-cache-dir transformers peft datasets accelerate scikit-learn pandas
COPY . /app
# Do NOT install an NVIDIA driver here; Fuzzball injects the host driver at runtime.
