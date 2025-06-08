provider "aws" {
  region = "eu-west-3"
}

resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # TODO: Restrict this to your IP for security
     }

  ingress {
    description = "WebApp HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Frontend Port 8080"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "docker_host" {
  ami                    = "ami-007c433663055a1cc"
  instance_type          = "t2.medium"
  key_name               = "key_mac_ed25519"
  vpc_security_group_ids = [aws_security_group.allow_ssh.id]

  user_data = file("install.sh")

  tags = {
    Name = "docker-compose-fastapi-web"
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("/Users/vauclinetienne/.ssh/id_ed25519")
    host        = self.public_ip
    timeout     = "10m"  # if timeout occurs,increase this
  }

  provisioner "remote-exec" {
    inline = [
      "echo 'Instance is ready'"
    ]
  }

  provisioner "file" {
    source      = "docker-compose.yml"
    destination = "/home/ubuntu/docker-compose.yml"
  }

  provisioner "file" {
    source      = "Dockerfile_ml"
    destination = "/home/ubuntu/Dockerfile_ml"
  }

  provisioner "file" {
    source      = "Dockerfile_web"
    destination = "/home/ubuntu/Dockerfile_web"
  }

  provisioner "file" {
    source      = "app"
    destination = "/home/ubuntu/app"
  }

  provisioner "file" {
    source      = "src"
    destination = "/home/ubuntu/src"
  }

  # Wait for cloud-init and install dependencies
  provisioner "remote-exec" {
    inline = [
      "echo 'Waiting for cloud-init to finish...'",
      "cloud-init status --wait",
      "echo 'Installing Docker and Docker Compose...'",
      "sudo apt-get update",
      "sudo apt-get install -y docker.io",
      "sudo systemctl start docker",
      "sudo systemctl enable docker",
      "sudo usermod -aG docker ubuntu",
      "sudo curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose",
      "sudo chmod +x /usr/local/bin/docker-compose",
      "echo 'Starting services...'",
      "cd /home/ubuntu",
      "sudo docker-compose up -d"
    ]
  }
}

output "web_url" {
  value = "http://${aws_instance.docker_host.public_ip}:8080"
}

output "fastapi_url" {
  value = "http://${aws_instance.docker_host.public_ip}"
}

output "ssh_command" {
  value = "ssh -i /Users/vauclinetienne/.ssh/id_ed25519 ubuntu@${aws_instance.docker_host.public_ip}"
}