pipeline {
    agent any
    
    environment {
        PROJECT_ID = 'legal-rag'
        REPOSITORY_NAME = 'test'
        GCR_URL = "asia.gcr.io/${PROJECT_ID}/${REPOSITORY_NAME}/test_image"
        SCAN_IMAGE = "anchore/scan-action:latest"
    }
    
    stages {
        stage('SCM Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/lily4499/lil-node-app.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${GCR_URL}:${BUILD_NUMBER}")
                }
            }
        }

        stage('Scan Docker Image for Vulnerabilities') {
            steps {
                script {
                    sh """
                        image --ignore-unfixed --scanners vuln --output ${BUILD_NUMBER}-vul.txt ${GCR_URL}:${BUILD_NUMBER}
                    """
                }
            }
        }

        stage('Manual Approval') {
            steps {
                script {
                    input message: 'Review the scan results and click "Proceed" to continue if there are no vulnerabilities.', ok: 'Proceed'
                }
            }
        }
        // stage('Push to GCR') {
        //     steps {
        //         script {
        //             docker.withRegistry('https://asia.gcr.io', 'google_registry') {
        //                 docker.image("${GCR_URL}:${BUILD_NUMBER}").push()
        //             }
        //         }
        //     }
        // }

        // stage('Deploy to Google Kubernetes Engine') {
        //     agent {
        //         kubernetes {
        //             containerTemplate {
        //                 name 'helm' // Name of the container to be used for helm upgrade
        //                 image 'asia.gcr.io/legal-rag/jenkins_helm:v0.1' // The image containing helm
        //             }
        //         }
        //     }
        //     steps {
        //         script {
        //             steps
        //             container('helm') {
        //                 sh("helm upgrade --install hpp2 ./helm --namespace model-serving")
        //             }
        //         }
        //     }
    }
}