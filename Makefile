.PHONY: shaders check lint fix install test test-fast test-file clean help completions

help:
	@echo "Available commands:"
	@echo "  make check   - Run type checker and linter"
	@echo "  make lint    - Run ruff linter only"
	@echo "  make fix     - Auto-fix linting issues and format"
	@echo "  make install - Install package in editable mode"
	@echo "  make test-file F=tests/test_foo.py - One file, serial (the iteration loop)"
	@echo "  make test-fast - Everything but the slow set (after an iteration)"
	@echo "  make test    - The whole non-GPU suite (two or three times a week)"
	@echo "  make clean   - Remove generated files"
	@echo "  make completions - Regenerate shell completions for the ffs_* tools"

check: lint
	ty check fastfuncstuff/

lint:
	ruff check fastfuncstuff/ examples/

fix:
	ruff check --fix fastfuncstuff/ examples/
	ruff format fastfuncstuff/ examples/

install:
	pip install -e ".[dev]"

# Three tiers, by how often they are meant to run. See the testing section of
# CLAUDE.md for the measured costs and for what `slow` means.
# One file, serially: the loop during iterative work. Most of its ~5 s is the
# torch import, so there is nothing to gain by narrowing further than a file.
test-file:
	@test -n "$(F)" || { echo "usage: make test-file F=tests/test_foo.py"; exit 2; }
	pytest $(F) -n 0

# Once a change is finished. -m REPLACES addopts' own -m, so `not gpu` has to be
# repeated here or every GPU test comes back.
test-fast:
	pytest tests/ -m "not slow and not gpu"

# The whole non-GPU set, slow tests included. Run it a couple of times a week:
# the slow set rots when nothing exercises it (it has already happened once).
test:
	pytest tests/

completions:
	@# Uses THIS environment's python on purpose: the tool list comes from its
	@# installed metadata, so generating from another env writes a partial set.
	python -m fastfuncstuff.cli.completion -shell fish -o $(HOME)/.config/fish/completions \
		-help_dir $(HOME)/.local/share/ffs/flag_help \
		-keybind $(HOME)/.config/fish/conf.d/ffs_flag_help.fish
	python -m fastfuncstuff.cli.completion -shell bash -o $(HOME)/.local/share/bash-completion/completions \
		-help_dir $(HOME)/.local/share/ffs/flag_help \
		-keybind $(HOME)/.local/share/ffs/flag_help/keybind.bash
	python -m fastfuncstuff.cli.completion -shell zsh  -o $(HOME)/.zfunc \
		-help_dir $(HOME)/.local/share/ffs/flag_help \
		-keybind $(HOME)/.local/share/ffs/flag_help/keybind.zsh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage .pytest_cache/ dist/ build/

# The 3-D surface view's shaders, baked for every RHI backend (GL, GLES,
# Vulkan/SPIR-V, D3D/HLSL, Metal/MSL). The .qsb files are committed so a user
# never needs the baker; rerun after editing a .vert/.frag.
SHADERS := fastfuncstuff/viewer/shaders
shaders:
	for f in $(SHADERS)/*.vert $(SHADERS)/*.frag; do \
		pyside6-qsb --glsl "330,300 es" --hlsl 50 --msl 12 -o $$f.qsb $$f || exit 1; \
	done
