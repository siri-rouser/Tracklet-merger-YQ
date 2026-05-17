# Tracklet Merger

## Functionality

This repository receives inter-camera tracklet data (as protobuf messages) from edge-node object trackers via Redis streams and performs multi-camera / cross-camera identity association. 

Results are written incrementally to `results/cross_camera_matches.jsonl`.

## Paper

This repository implements the cross-camera association module described in:\
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6626356>

## Set Up

**Option 1 — Poetry (virtual environment)**
```bash
poetry install
python main.py   # configure settings.yaml first
```

**Option 2 — Docker**
```bash
bash docker_build.sh
```

## Other Modules

Other modules and the full operation pipeline from the paper will be partially released in the future.

## SAE Engine

This repository is developed from <https://github.com/starwit/sae-stage-template>.

The end-to-end workflow described in the paper is built on top of the [Starwit Awareness Engine](https://github.com/starwit/starwit-awareness-engine).

## Citataion

If you use this work, please cite:

```bibtex
@article{lin2025edge,
  title     = {Edge Assisted Multi-Camera Vehicle Tracking Framework for Real-Time and Scalable Deployment},
  author    = {Lin, Yuqiang and Lockyer, Sam and Zhang, Shucheng and Stanek, Florian and Zarbock, Markus and Evans, Adrian and Li, Wenbin and Wang, Yinhai and Zhang, Nic},
  journal   = {SSRN Electronic Journal},
  year      = {2025},
  doi       = {10.2139/ssrn.6626356},
  url       = {https://ssrn.com/abstract=6626356}
}
```

---

### License

AGPLv3 — see [LICENSE](LICENSE).
