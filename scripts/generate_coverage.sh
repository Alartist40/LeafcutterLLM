#!/bin/bash
# generate_coverage.sh - Generate test coverage report

set -e

echo "🧪 Running tests and generating coverage report..."
go test ./... -cover -coverprofile=coverage.out
go tool cover -func=coverage.out

echo ""
echo "📊 Generating HTML report..."
go tool cover -html=coverage.out -o coverage.html

echo "✅ Done. HTML report saved to: coverage.html"
echo "   Total coverage: $(go tool cover -func=coverage.out | grep total | awk '{print $3}')"
