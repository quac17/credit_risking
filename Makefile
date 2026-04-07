# Pipeline Docker — chạy từ thư mục gốc repo (cùng cấp docker-compose.yml).
# Trên Windows: dùng GNU Make (Git Bash, MSYS2) hoặc chạy trực tiếp lệnh docker compose trong README.

.PHONY: help
.PHONY: docker-build docker-shell docker-woe docker-simplified docker-sample
.PHONY: docker-train-softmax docker-train-softmax-extra
.PHONY: docker-validate-softmax docker-test-softmax
.PHONY: docker-train-coral docker-validate-coral docker-test-coral

# Tham số thêm cho train Softmax (vd: make docker-train-softmax-extra EXTRA_ARGS="--epochs 20 --train-sampler shuffle --focal-gamma 0 --class-4-sample-weight 1")
EXTRA_ARGS ?=

help:
	@echo "Docker Compose — credit_risking"
	@echo ""
	@echo "  make docker-build              # build image"
	@echo "  make docker-shell              # bash trong container"
	@echo "  make docker-woe                # IV/WoE"
	@echo "  make docker-simplified         # create_simplified_data"
	@echo "  make docker-sample             # sample_data"
	@echo ""
	@echo "  make docker-train-softmax      # train Softmax — mặc định ưu tiên nhãn lớp cuối (config.py)"
	@echo "  make docker-train-softmax-extra EXTRA_ARGS=\"...\"   # ghi đè / train kiểu cũ (vd. shuffle+CE thuần)"
	@echo ""
	@echo "  make docker-validate-softmax | docker-test-softmax"
	@echo "  make docker-train-coral | docker-validate-coral | docker-test-coral"
	@echo ""

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

docker-train-softmax-extra:
	docker compose run --rm train-mlp-softmax python deep_credit_rating/mlp_softmax/train.py $(EXTRA_ARGS)

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
