# Tracklet Merger 

This repositry is developed based on https://github.com/starwit/sae-stage-template. 

# Functionality

The main goal of this repo is receive inter-camera processed Redis message from other edge-nodes and performance multi-camera/cross camera association.

# Paper

This repo corresponds to the cross-camera assocation module in paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6626356

# Set up 

simply use poetry install to set up the enviornment. cofiguration are in settings.yaml, config all hyperparameters before using python main.py to run the whole pipeline.

or use dockerfile to build docker images.

# Other modules

Other modules from the paper 

---

## License

AGPLv3 — see [LICENSE](LICENSE).
If you made a mistake either reclone the repository or run `git reset --hard && git clean -fd`
