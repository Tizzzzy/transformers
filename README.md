1. conda create -n moe python==3.12
2. conda activate moe
3. git clone https://github.com/Tizzzzy/transformers.git
4. cd transformers
5. pip install "transformers[torch]"
6. sudo apt update
7. sudo apt install make
8. pip install ruff libcst rich
9. make fix-copies
10. find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +
11. pip uninstall transformers
12. pip install -e .