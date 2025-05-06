cvt_docker() {
    local usage_str="USAGE: cvt_docker [start|stop|logs] [--dev|--prd]"
    local docker_path=""
    local current_env=""

    case "$2" in
        --dev|-d|dev)
            current_env="dev"
            ;;
        --prd|-p|prd)
            current_env="prd"
            ;;
        "")
            current_env=${CVT_CURRENT_ENV:-"prd"}
            ;;
        *)
            printf "Error: Invalid argument\n$usage_str\n"
            return 0
            ;;
    esac

    case "$current_env" in
        dev)
            docker_path="docker-compose-dev.yml"
            ;;
        prd)
            docker_path="docker-compose.yml"
            ;;
    esac

    if [ -z ${CVT_WORK_DIR+x} ]; then 
        echo "CVT_WORK_DIR not set, exiting"
        return 0
    else
        docker_path="$CVT_WORK_DIR/$docker_path"
    fi

    case "$1" in
        start)
            export CVT_CURRENT_ENV=$current_env
            docker compose --file $docker_path up -d --build
            ;;
        stop)
            unset CVT_CURRENT_ENV
            docker compose --file $docker_path down
            ;;
        logs)
            docker compose --file $docker_path logs -f
            ;;
        *)
            printf "Error: Invalid argument\n$usage_str\n"
            return 0
            ;;
    esac
}