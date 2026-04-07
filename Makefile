.PHONY: docker-build docker-shell docker-woe docker-simplified docker-sample
.PHONY: docker-train-softmax docker-validate-softmax docker-test-softmax
.PHONY: docker-train-coral docker-validate-coral docker-test-coral

docker-build:
	docker compose build

docker-shell:
	docker compose run --rm app bash

docker-woe:
	docker compose run --rm woe-analysis

docker-simplified:
	docker compose run --rm create-simplified

docker-sample:
	docker compose run --rm sample-data

docker-train-softmax:
	docker compose run --rm train-mlp-softmax

docker-validate-softmax:
	docker compose run --rm validate-mlp-softmax

docker-test-softmax:
	docker compose run --rm test-mlp-softmax

docker-train-coral:
	docker compose run --rm train-mlp-coral

docker-validate-coral:
	docker compose run --rm validate-mlp-coral

docker-test-coral:
	docker compose run --rm test-mlp-coral
