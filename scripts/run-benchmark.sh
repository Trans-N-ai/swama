#!/bin/bash

# AI Model Speed Comparison Test Script
# Compare response speed between ollama and swama

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_PROMPT="Write a whimsical and imaginative fairy tale that introduces the concept of quantum gravity in a way that is accessible to children or curious readers. Set the story in a fantastical universe where the laws of physics are alive, personified, and full of character. The main character could be a curious young explorer, a talking particle, or a tiny wizard who lives between atoms. In this world, classical gravity (embodied by an old and wise King Newton) rules over the visible realm—the mountains, rivers, planets, and stars. But strange things begin to happen at the tiniest scales—inside atoms, near magical black holes, or in a mysterious “String Forest.” The protagonist discovers that there is a secret layer of reality governed by quantum rules, full of uncertainty, superpositions, and entangled creatures. Introduce a character who represents quantum gravity—perhaps a mischievous but brilliant child named Quanta, who is trying to unify King Newton’s kingdom with the chaotic, dance-like domain of Queen Quantum. Let the tale unfold as a journey to reconcile the two realms, filled with magical metaphors for wave functions, spacetime foam, graviton sprites, and Planck-scale riddles. The story should follow a classic fairy tale structure: a curious quest, a conflict between opposing forces or worldviews, magical helpers or obstacles, and a resolution that brings harmony or a profound lesson. Use rich and poetic language, playful imagery, and occasional gentle humor. The goal is not to teach equations but to enchant the imagination while hinting at the wonders of quantum gravity."
TEST_ROUNDS=3
SWAMA_URL="http://127.0.0.1:28100/v1/chat/completions"

# Model mapping configuration (format: "display_name|ollama_name|swama_name")
MODELS=(
    "Qwen3-1.7B|qwen3:1.7b|mlx-community/Qwen3-1.7B-4bit"
    "Qwen3-4B|qwen3:4b|mlx-community/Qwen3-4B-4bit"
    "Qwen3-8B|qwen3:8b|mlx-community/Qwen3-8B-4bit"
    "Qwen3-14B|qwen3:14b|mlx-community/Qwen3-14B-4bit"
    # "Qwen3-30B|qwen3:30b|mlx-community/Qwen3-30B-A3B-4bit"
    # "Qwen3-32B|qwen3:32b|mlx-community/Qwen3-32B-4bit"
    # "Qwen3-235B|qwen3:235b|mlx-community/Qwen3-235B-A22B-4bit"
    "DeepSeek-R1-8B|deepseek-r1:8b|mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"
    # "DeepSeek-R1-671B|deepseek-r1:671b|mlx-community/DeepSeek-R1-0528-4bit"
)

echo -e "${BLUE}=== AI Model Speed Comparison Test ===${NC}"
echo -e "Test prompt: ${GREEN}\"${TEST_PROMPT}\"${NC}"
echo -e "Test rounds: ${GREEN}${TEST_ROUNDS}${NC}"
echo -e "Number of models: ${GREEN}${#MODELS[@]}${NC}"
echo ""

# Check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 command not found${NC}"
        return 1
    fi
    return 0
}

# Test function - Ollama
test_ollama() {
    local model_name="$1"
    local total_tokens_per_sec=0
    local success_count=0
    
    echo -e "${YELLOW}Testing Ollama (${model_name})...${NC}"
    
    for i in $(seq 1 $TEST_ROUNDS); do
        echo -n "  Round $i/$TEST_ROUNDS: "
        
        # Create temporary file to store output
        temp_file=$(mktemp)
        
        # Execute ollama command and capture output
        if ollama run "$model_name" "$TEST_PROMPT" --verbose > "$temp_file" 2>&1; then
            # Extract eval rate from ollama output
            tokens_per_sec=$(grep "eval rate:" "$temp_file" | tail -1 | grep -o '[0-9]*\.[0-9]*' | head -1)
            
            if [ -n "$tokens_per_sec" ] && [ "$tokens_per_sec" != "0" ]; then
                total_tokens_per_sec=$(echo "$total_tokens_per_sec + $tokens_per_sec" | bc)
                success_count=$((success_count + 1))
                
                echo -e "${GREEN}${tokens_per_sec} tokens/s${NC}"
            else
                echo -e "${RED}Failed to get speed data${NC}"
            fi
        else
            echo -e "${RED}Failed${NC}"
        fi
        
        rm -f "$temp_file"
    done
    
    if [ $success_count -gt 0 ]; then
        avg_tokens_per_sec=$(echo "scale=2; $total_tokens_per_sec / $success_count" | bc)
        echo -e "  Ollama average speed: ${GREEN}${avg_tokens_per_sec} tokens/s${NC} (success $success_count/$TEST_ROUNDS times)"
        # Only output the number for capture
        echo "$avg_tokens_per_sec"
    else
        echo -e "  Ollama test failed: ${RED}All tests failed${NC}"
        # Only output 0 for capture
        echo "0"
    fi
}

# Test function - Swama
test_swama() {
    local model_name="$1"
    local total_tokens_per_sec=0
    local success_count=0
    
    echo -e "${YELLOW}Testing Swama (${model_name})...${NC}"
    
    for i in $(seq 1 $TEST_ROUNDS); do
        echo -n "  Round $i/$TEST_ROUNDS: "
        
        # Create temporary file to store response
        temp_file=$(mktemp)
        
        # Execute curl command
        if curl --no-buffer -s "$SWAMA_URL" \
          -H "Content-Type: application/json" \
          -d "{\"model\":\"$model_name\",\"messages\":[{\"role\":\"user\",\"content\":\"$TEST_PROMPT\"}],\"stream\":true}" \
          > "$temp_file" 2>&1; then
            grep -o 'response_token' "$temp_file"
            tokens_per_sec=$(grep -o '"response_token\\/s":[0-9.]*' "$temp_file" | tail -1 | cut -d':' -f2)
                        
            if [ -n "$tokens_per_sec" ] && [ "$tokens_per_sec" != "0" ]; then
                total_tokens_per_sec=$(echo "$total_tokens_per_sec + $tokens_per_sec" | bc)
                success_count=$((success_count + 1))
                
                echo -e "${GREEN}${tokens_per_sec} tokens/s${NC}"
            else
                echo -e "${RED}Failed to get speed data${NC}"
            fi
        else
            echo -e "${RED}Failed${NC}"
        fi
        
        rm -f "$temp_file"
    done
    
    if [ $success_count -gt 0 ]; then
        avg_tokens_per_sec=$(echo "scale=2; $total_tokens_per_sec / $success_count" | bc)
        echo -e "  Swama average speed: ${GREEN}${avg_tokens_per_sec} tokens/s${NC} (success $success_count/$TEST_ROUNDS times)"
        echo "$avg_tokens_per_sec"
    else
        echo -e "  Swama test failed: ${RED}All tests failed${NC}"
        echo "0"
    fi
}

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
ollama_available=true
swama_available=true

if ! check_command "ollama"; then
    ollama_available=false
    echo -e "${YELLOW}Warning: ollama command not found${NC}"
fi

if ! check_command "curl"; then
    swama_available=false
    echo -e "${YELLOW}Warning: curl command not found${NC}"
fi

if ! check_command "bc"; then
    echo -e "${RED}Error: bc command not found, please install bc for calculations${NC}"
    exit 1
fi

echo ""

# Store results for table output (using simple variables instead of associative arrays)
ollama_results=""
swama_results=""

# Test each model
for model_config in "${MODELS[@]}"; do
    # Parse model configuration
    IFS='|' read -ra model_info <<< "$model_config"
    display_name="${model_info[0]}"
    ollama_model="${model_info[1]}"
    swama_model="${model_info[2]}"
    
    echo -e "${BLUE}=== Testing Model: ${display_name} ===${NC}"
    echo ""
    
    # Test Ollama
    if [ "$ollama_available" = true ]; then
        ollama_speed=$(test_ollama "$ollama_model" 2>/dev/null | tail -1)
        ollama_results="${ollama_results}${display_name}:${ollama_speed};"
        
        # Kill ollama after each model test
        echo "  Stopping ollama service..."
        pkill -f ollama || true
        sleep 2
    else
        echo -e "${YELLOW}Skip Ollama test (command not found)${NC}"
        ollama_results="${ollama_results}${display_name}:N/A;"
    fi
    
    echo ""
    
    # Test Swama
    if [ "$swama_available" = true ]; then
        echo "  Starting swama serve..."
        swama serve &
        swama_pid=$!
        sleep 5  # Wait for service to start
        
        swama_speed=$(test_swama "$swama_model" 2>/dev/null | tail -1)
        swama_results="${swama_results}${display_name}:${swama_speed};"
        
        # Kill swama after each model test
        echo "  Stopping swama service..."
        kill $swama_pid 2>/dev/null || true
        pkill -f "swama serve" || true
        sleep 2
    else
        echo -e "${YELLOW}Skip Swama test (command not found)${NC}"
        swama_results="${swama_results}${display_name}:N/A;"
    fi
    
    echo ""
done

# Output results table
echo -e "${BLUE}=== Performance Summary Table ===${NC}"
echo ""
printf "| %-15s | %-20s | %-20s |\n" "Model" "Ollama (tokens/s)" "Swama (tokens/s)"
printf "| %-15s- | -%-20s- | -%-20s |\n" "---------------" "--------------------" "--------------------"

for model_config in "${MODELS[@]}"; do
    IFS='|' read -ra model_info <<< "$model_config"
    display_name="${model_info[0]}"
    
    # Extract results from stored strings
    ollama_val=$(echo "$ollama_results" | grep -o "${display_name}:[^;]*" | cut -d':' -f2)
    swama_val=$(echo "$swama_results" | grep -o "${display_name}:[^;]*" | cut -d':' -f2)
    
    # Format values (show 0 as "Failed")
    if [ "$ollama_val" = "0" ]; then
        ollama_display="Failed"
    else
        ollama_display="$ollama_val"
    fi
    
    if [ "$swama_val" = "0" ]; then
        swama_display="Failed"
    else
        swama_display="$swama_val"
    fi
    
    printf "| %-15s | %-20s | %-20s |\n" "$display_name" "$ollama_display" "$swama_display"
done

echo ""
echo -e "${BLUE}Test completed!${NC}"
