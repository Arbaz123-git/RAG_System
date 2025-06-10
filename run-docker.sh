#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi

# Function to check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}Warning: .env file not found.${NC}"
        echo "Creating a template .env file. Please edit it with your actual API keys."
        cat > .env << EOL
# API Keys
GROQ_API_KEY=your_groq_api_key_here

# JWT Settings
JWT_SECRET_KEY=your_jwt_secret_key_for_production_use_a_strong_random_string

# Cache Settings
MEMORY_CACHE_SIZE=5000
REDIS_TTL=86400
EOL
        echo -e "${YELLOW}Created .env file. Please edit it with your actual API keys before continuing.${NC}"
        echo "Press Enter to continue or Ctrl+C to exit and edit the file..."
        read
    fi
}

# Function to start the services
start_services() {
    echo -e "${GREEN}Starting MultiModal RAG services...${NC}"
    docker-compose up -d
    
    echo -e "${GREEN}Checking service status...${NC}"
    docker-compose ps
    
    echo -e "${GREEN}Services started successfully!${NC}"
    echo -e "API is available at: ${YELLOW}http://localhost:8000${NC}"
    echo -e "To get a token, run: ${YELLOW}curl -X POST \"http://localhost:8000/token\" -H \"Content-Type: application/x-www-form-urlencoded\" -d \"username=clinician1&password=secret1\"${NC}"
}

# Function to stop the services
stop_services() {
    echo -e "${GREEN}Stopping MultiModal RAG services...${NC}"
    docker-compose down
    echo -e "${GREEN}Services stopped successfully!${NC}"
}

# Function to show logs
show_logs() {
    echo -e "${GREEN}Showing logs for all services (press Ctrl+C to exit)...${NC}"
    docker-compose logs -f
}

# Function to run load tests
run_load_tests() {
    # Check if Locust service is uncommented in docker-compose.yml
    if grep -q "#.*locust:" docker-compose.yml; then
        echo -e "${YELLOW}Locust service is commented out in docker-compose.yml.${NC}"
        echo "Uncommenting it for you..."
        sed -i 's/# *locust:/  locust:/g' docker-compose.yml
        sed -i 's/#   build:/    build:/g' docker-compose.yml
        sed -i 's/#   ports:/    ports:/g' docker-compose.yml
        sed -i 's/#     - /      - /g' docker-compose.yml
        sed -i 's/#   volumes:/    volumes:/g' docker-compose.yml
        sed -i 's/#   environment:/    environment:/g' docker-compose.yml
        sed -i 's/#   depends_on:/    depends_on:/g' docker-compose.yml
        sed -i 's/#   networks:/    networks:/g' docker-compose.yml
        sed -i 's/#   command:/    command:/g' docker-compose.yml
    fi
    
    echo -e "${GREEN}Starting Locust for load testing...${NC}"
    docker-compose up -d locust
    
    echo -e "${GREEN}Locust UI is available at: ${YELLOW}http://localhost:8089${NC}"
    echo "Open this URL in your browser to configure and start the load test."
}

# Show menu
show_menu() {
    echo -e "${GREEN}=== MultiModal RAG Docker Management ===${NC}"
    echo "1) Start services"
    echo "2) Stop services"
    echo "3) Show logs"
    echo "4) Run load tests"
    echo "5) Exit"
    echo -n "Enter your choice [1-5]: "
    read choice
    
    case $choice in
        1)
            check_env_file
            start_services
            ;;
        2)
            stop_services
            ;;
        3)
            show_logs
            ;;
        4)
            run_load_tests
            ;;
        5)
            echo -e "${GREEN}Exiting.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            ;;
    esac
    
    echo ""
    show_menu
}

# Main
echo -e "${GREEN}Welcome to MultiModal RAG Docker Management${NC}"
show_menu 