FROM astral/uv:python3.14-bookworm-slim

RUN <<EOT
  /usr/local/bin/uv venv -p 3.14
  . .venv/bin/activate
  /usr/local/bin/uv pip install --link-mode=copy --no-cache pandas nicegui isort matplotlib xlrd openpyxl pyxlsb
  groupadd --system --gid 999 nonroot
  useradd --system --gid 999 --uid 999 --create-home nonroot
EOT

WORKDIR /app

USER nonroot

ENTRYPOINT ["/.venv/bin/python3", "main.py"]
# ENTRYPOINT ["/bin/bash"]
