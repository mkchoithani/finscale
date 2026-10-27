# FinScale

Financial-risk classifier: LoRA fine-tuning of Qwen2.5-0.5B-Instruct, intended for a 1-vs-2 GB10-node benchmark and OpenAI-compatible serving. Hardware, image, and serving compatibility must be verified in the event environment.

## Local smoke test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python gen_data.py
python train_gb10.py --epochs 1 --bs 2
```

The local workflow is not a validation of CUDA, arm64, or GB10 behavior.

## Fuzzball

Fuzzball examples are in `fuzzball/single_node.fz`, `fuzzball/multi_node.fz`, and `fuzzball/smoke_test.fz`. They use the documented Fuzzfile keys and deliberately retain the image URI placeholder:

```text
docker://<REGISTRY>/<IMAGE>:<TAG>
```

Do not submit the placeholders unchanged. The single-node workflow runs training, starts a `serve` service after training completion, and runs `eval` after `serve` enters RUNNING. RUNNING does not guarantee readiness: the service has a TCP readiness probe and evaluation retries the OpenAI-compatible `/v1/models` endpoint before calling `evaluate_endpoint.py`. Confirm the service DNS hostname and the documented completion-status string (currently marked TODO in the Fuzzfile) with organizers.

### Ask organizers

- What is the team namespace?
- What is the approved image publishing method?
- How are registry credentials provided to jobs/builds?
- Are `fb://` objects used in this event? `fb://` refers to Fuzzball Object Cache, not a container registry; it is not a substitute for the `docker://` image URI.
- Which PyTorch NGC base tag has an arm64 manifest and is compatible with the event GB10 and host driver?
- What are the service hostname convention, dependency completion-status value, multi-node SSH-wrapper invocation, and host-list format for the event version?

No registry or image tag is presumed here. Build only after organizers confirm a base image and publishing process. For example, set the approved base image explicitly at build time:

```bash
docker buildx build --platform linux/arm64 \\
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:<VERIFIED_TAG> \\
  -t <REGISTRY>/<IMAGE>:<TAG> .
```

The tag `25.09-py3` is unverified for this purpose. Verify the chosen tag's arm64 manifest (newer tags such as `26.09-py3` list arm64 in NGC) and GB10/driver compatibility. Run `fuzzball/smoke_test.fz` before training to check the allocated PyTorch CUDA runtime. It reports the PyTorch version, CUDA availability, device name/capability, and a small GPU matrix multiplication.

## Multi-node

The generic multi-node job uses the documented environment variables `MULTINODE_NODE_IP`, `MULTINODE_HOSTLIST_NOSLOTS`, and `MULTINODE_SSH_WRAPPER`. `scripts/launch_multinode.sh` derives node ranks from the host list, uses `${MULTINODE_NODE_IP}` as `--master-addr`, launches rank 0 locally and worker ranks through the SSH wrapper. The wrapper command syntax and host-list format are version/event-specific and explicitly TODO for organizer confirmation. This flow is not yet verified in an event environment.

## Serving image

The serve script accepts `SERVER_CMD`. Its commented default is a vLLM OpenAI server command and is **UNTESTED on arm64/GB10**:

```bash
python -m vllm.entrypoints.openai.api_server --model /models/merged --host 0.0.0.0 --port 8000
```

No inference engine is selected as confirmed. Alternatives to evaluate with organizers include llama.cpp and TensorRT-LLM; confirm arm64/GB10 availability, model format, and runtime support before choosing one. `SERVER_CMD` can be set to the approved alternative command.

## Image build

```bash
docker buildx build --platform linux/arm64 \\
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:<VERIFIED_TAG> \\
  -t <REGISTRY>/<IMAGE>:<TAG> .
```

Registry namespace, URI, credentials, and publishing steps remain organizer-provided. No push command is assumed.
