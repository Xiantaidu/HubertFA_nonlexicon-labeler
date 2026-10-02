# -*- coding: utf-8 -*-
"""Extract the nonlexicon labeler (CVNT branch) from a merged HubertFA model.onnx.

The merged HubertFA ONNX (exported by scripts/onnx_exporter.py, ONNX_EXPORT_VERSION 5)
is a composition of encoder + predict models with inputs:
    waveform -> outputs: [cvnt_logits, ph_frame_logits, ph_edge_logits, ctc_logits]

This script slices the subgraph feeding `cvnt_logits` (the nonlexicon labeler /
CVNT branch) into a standalone ONNX via onnx.utils.extract_model. The subgraph
keeps the Hubert encoder because the CVNT branch only consumes the aligned
encoder features, not the raw waveform.

Usage:
    python extract_nonlexicon.py --model model.onnx --out nonlexicon_labeler.onnx

Note: run with an env that has `onnx` installed (training env, see main repo
requirements_onnx.txt). onnxruntime-only envs cannot run this.
"""
import argparse

from onnx.utils import extract_model


def main():
    parser = argparse.ArgumentParser(description="Extract nonlexicon labeler subgraph from merged HubertFA ONNX")
    parser.add_argument("--model", required=True, help="path to merged model.onnx")
    parser.add_argument("--out", required=True, help="path to write nonlexicon_labeler.onnx")
    args = parser.parse_args()

    extract_model(args.model, args.out, ["waveform"], ["cvnt_logits"])

    import onnx
    m = onnx.load(args.out)
    print(f"saved: {args.out}")
    print(f"  inputs:  {[i.name for i in m.graph.input]}")
    print(f"  outputs: {[o.name for o in m.graph.output]}")
    print(f"  nodes:   {len(m.graph.node)}")


if __name__ == "__main__":
    main()
