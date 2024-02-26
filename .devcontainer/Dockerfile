FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set the network mode
# Note: "host" network mode is not directly supported in Dockerfiles.
#       You may need to handle networking externally or use different approaches.
#       This line is added to show the equivalent Dockerfile command.
#       The behavior might not be the same as using "--network host" in run command.
# See Docker networking documentation for alternatives: https://docs.docker.com/network/
# This is done in  .devcontainer/devcontainer.json


# Update package lists and install necessary tools
RUN apt-get update && apt-get install -y wget openssh-client git-core bash-completion

# # Install git-lfs
RUN wget -O /tmp/git-lfs.deb https://packagecloud.io/github/git-lfs/packages/ubuntu/focal/git-lfs_2.13.3_amd64.deb/download.deb && \
    dpkg -i /tmp/git-lfs.deb && \
    rm /tmp/git-lfs.deb

RUN echo  'source /usr/share/bash-completion/completions/git' >> ~/.bashrc 


# Command to run on container startup
CMD ["/bin/bash"]
