pipeline {
    agent any
    stages {
        stage("Checkout") {
            steps {
                checkout scm
                sh "mkdir -p var/static var/media var/log"
                sh "rm -rf .venv"
                sh "echo 'COMPOSE_PROJECT_NAME=${env.JOB_NAME}-${env.BUILD_ID}' > .env"
                sh "docker --version; docker-compose --version"
            }
        }
        stage("Build") {
            steps {
                sh "docker-compose down --volumes"
                sh "docker-compose build --build-arg uid=`id -u` --build-arg gid=`id -g` lib"
                sh "docker-compose run --rm --volume $HOME/.netrc:/home/nens/.netrc --volume $HOME/.cache/pip:/home/nens/.cache/pip --volume $HOME/.cache/pipenv:/home/nens/.cache/pipenv lib pipenv sync --dev"
                sh "docker-compose run --rm lib pip freeze"
            }
        }
        stage("Test") {
            steps {
                sh "docker-compose run --rm lib pipenv check"
                sh "docker-compose run --rm lib pipenv run nosetests"
            }
        }
    }
    post {
        always {
            sh "docker-compose down --volumes --remove-orphans && docker-compose rm -f && rm -rf .venv"
        }
    }
}
