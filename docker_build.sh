#!/bin/bash

docker build -t starwitorg/sae-my-stage:$(poetry version --short) .