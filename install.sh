#!/bin/bash
set -e

apt-get update

apt-get install -y docker.io

systemctl start docker
systemctl enable docker

usermod -aG docker ubuntu

curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# apt-get install -y docker-compose-plugin

echo "Docker and Docker Compose installation completed"