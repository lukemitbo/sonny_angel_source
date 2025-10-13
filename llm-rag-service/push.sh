
# 1) (once) create the repo if it doesn't exist
aws ecr describe-repositories --repository-names llm-rag-service --region $AWS_REGION \
  || aws ecr create-repository --repository-name llm-rag-service --region $AWS_REGION

if [ -z "$AWS_REGION" ]; then
  echo "AWS_REGION is not set"
  exit 1
fi

if [ -z "$AWS_ACCOUNT_ID" ]; then
  echo "AWS_ACCOUNT_ID is not set"
  exit 1
fi

# 2) login to your private ECR registry
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# 3) build & push (choose the platform you intend to run on)
export IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/llm-rag-service:$(git rev-parse --short HEAD)

docker buildx create --use >/dev/null 2>&1 || true
# for ECS x86_64:
# docker buildx build --platform linux/amd64 -t $IMAGE --push .
# or for ECS ARM64:
docker buildx build --platform linux/arm64/v8 -t $IMAGE --push .

echo "Pushed: $IMAGE"
