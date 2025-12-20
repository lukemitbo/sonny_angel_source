resource "aws_lb" "my_alb" {
  name = "my-alb"
  load_balancer_type = "application"
  subnet_mapping {
    subnet_id = "subnet-000a90618934e334e"
  } 
  subnet_mapping {
    subnet_id = "subnet-01e902732ff7766f4"
  }
}

