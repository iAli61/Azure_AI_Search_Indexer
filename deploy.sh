#!/bin/bash

resourceGroupName="aimdemo"
location="westeurope"
registryName="aimdemoregistry"
appName="aimdemoapp"
version="v3"
appServicePlanName="aimdemoappserviceplan"
subscriptionId="f804f2da-c27b-45ac-bf80-16d4d331776d"
containerName="aimdemoappcontainer"

# Login to Azure
az login

# Set the active subscription
az account set --subscription $subscriptionId

# Create a resource group
# az group create --name $resourceGroupName --location $location 

# Check if the Azure Container Registry already exists
registryExists=$(az acr show --name $registryName --resource-group $resourceGroupName --query "name" -o tsv)

# Create an Azure Container Registry if it doesn't exist
if [ -z "$registryExists" ]; then
    az acr create --name $registryName --resource-group $resourceGroupName --sku standard --admin-enabled true --tags demo

    # wait for the Azure Container Registry to be enabled
    az acr wait --name $registryName --sku standard

fi

# get the Azure Container Registry uername and password

acrUsername=$(az acr credential show --name $registryName --resource-group $resourceGroupName --query "username" -o tsv)
acrPassword=$(az acr credential show --name $registryName --resource-group $resourceGroupName --query "passwords[0].value" -o tsv)

# login to the Azure Container Registry
az acr login --name $registryName --expose-token


# Build and push the Docker image to the Azure Container Registry
az acr build --registry $registryName --image $registryName.azurecr.io/$appName:$version .

# Check if the Azure Container Instance already exists delete it if it does
containerExists=$(az container show --name $containerName --resource-group $resourceGroupName --query "name" -o tsv)

if [ ! -z "$containerExists" ]; then
    az container delete --name $containerName --resource-group $resourceGroupName --yes
fi


# Create a container instance
az container create --resource-group $resourceGroupName --name $containerName \
    --image $registryName.azurecr.io/$appName:$version \
    --dns-name-label $containerName \
    --cpu 1 --memory 1 --ip-address Public --ports 8501 \
    --registry-login-server $registryName.azurecr.io \
    --registry-username $acrUsername --registry-password $acrPassword 



# show the container instance
az container show --name $containerName --resource-group $resourceGroupName --query "{FQDN:ipAddress.fqdn,ProvisioningState:provisioningState}" -o table

# pull the logs from the container instance
# az container logs --name $containerName --resource-group $resourceGroupName

# az appservice plan create --name $appServicePlanName --resource-group $resourceGroupName --sku S1 --is-linux
# az webapp create --resource-group $resourceGroupName --plan $appServicePlanName --name $appName --deployment-container-image-name $registryName.azurecr.io/$appName:v2
# az webapp config appsettings set --resource-group $resourceGroupName --name $appName --settings WEBSITES_PORT=8501


# # Create a container instance
# az container create --resource-group $resourceGroupName --name $containerName --image $appName:$version --dns-name-label $containerName --ports 80
