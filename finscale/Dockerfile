# Build with an organizer-approved image registry/tag after verifying the base image manifest.
# Verify arm64 manifest availability (newer NGC tags such as 26.09-py3 list arm64)
# and compatibility with the event GB10 hardware and host driver.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:<VERIFIED_TAG>
FROM ${BASE_IMAGE}
WORKDIR /app
RUN pip install --no-cache-dir transformers peft datasets accelerate scikit-learn pandas
COPY . /app
# Do NOT install an NVIDIA driver here; Fuzzball injects the host driver at runtime.
