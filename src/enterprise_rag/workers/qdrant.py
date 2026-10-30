"""One bounded native worker poll using the explicitly configured Qdrant backend."""

import argparse
import json
from dataclasses import asdict

from enterprise_rag.api.qdrant_demo import services_from_env
from enterprise_rag.workers.lite import LiteWorker
from enterprise_rag.workers.text import LiteTextStageExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--worker-id", default="qdrant-local-worker")
    args = parser.parse_args()
    services = services_from_env(initialize_collection=args.initialize_only)
    try:
        if not args.initialize_only:
            result = LiteWorker(services.metadata, args.worker_id).run_once(
                LiteTextStageExecutor(
                    services.metadata,
                    services.object_store,
                    dense_publisher=services.semantic.external_dense,
                )
            )
            print(json.dumps(asdict(result)))
            return 0 if result.completed or not result.claimed else 1
        return 0
    finally:
        services.semantic.external_dense.backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
