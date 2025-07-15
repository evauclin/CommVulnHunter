provider "aws" {
  region = "eu-west-3"
}
/*resource "aws_key_pair" "main" {
  key_name   = "windows-ed25519-key"
  public_key = file("C:/Users/farin/.ssh/id_ed25519.pub")
}*/

# VPC Configuration
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "docker-compose-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "docker-compose-igw"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "eu-west-3a"
  map_public_ip_on_launch = true

  tags = {
    Name = "docker-compose-public-subnet"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "docker-compose-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh_new"
  description = "Allow SSH inbound traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # TODO: Restrict this to your IP for security
  }

  ingress {
    description = "FastAPI Port"
    from_port   = 8000
    to_port     = 8000
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

  ingress {
    description = "Auth Service Port"
    from_port   = 9000
    to_port     = 9000
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
#  key_name               =  aws_key_pair.main.key_name
  key_name               = "key_mac_ed25519"
  vpc_security_group_ids = [aws_security_group.allow_ssh.id]
  subnet_id              = aws_subnet.public.id

  user_data = file("install.sh")

  root_block_device {
    volume_size           = 15
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name = "docker-compose-fastapi-web"
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    #private_key = file("C:/Users/farin/.ssh/id_ed25519")
    private_key = file("/Users/vauclinetienne/.ssh/id_ed25519")
    host        = self.public_ip
    timeout     = "10m" # if timeout occurs, increase this
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
    source      = "Dockerfile_auth"
    destination = "/home/ubuntu/Dockerfile_auth"
  }

  provisioner "file" {
    source      = "app"
    destination = "/home/ubuntu/app"
  }

  provisioner "file" {
    source      = "src"
    destination = "/home/ubuntu/src"
  }

  provisioner "file" {
    source      = "auth_system"
    destination = "/home/ubuntu/auth_system"
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
  value = "http://${aws_instance.docker_host.public_ip}:8000"
}


output "ssh_command" {
  //value = "ssh -i C:/Users/farin/.ssh/id_ed25519 ubuntu@${aws_instance.docker_host.public_ip}"
  value = "ssh -i ~/.ssh/id_ed25519 ubuntu@${aws_instance.docker_host.public_ip}"
}
