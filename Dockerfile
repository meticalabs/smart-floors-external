FROM anyscale/ray:2.44.1-slim-py310

COPY --chown=ray:ray . /home/ray/app

RUN echo "export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook" >> /home/ray/.bashrc

ENV GIT_TERMINAL_PROMPT=0
ARG GITHUB_TOKEN
RUN echo "machine github.com login x-oauth-token password $GITHUB_TOKEN" > ~/.netrc
RUN chmod 600 ~/.netrc

WORKDIR /home/ray/app

RUN pip install uv

# Verify uv installation
RUN uv --version

# Install the project and its dependencies from pyproject.toml
RUN uv pip install --system --no-cache-dir -e .