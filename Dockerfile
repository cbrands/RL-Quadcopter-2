FROM continuumio/anaconda3

## make a local directory
RUN mkdir /opt/app

# set "/opt/app" as the working directory from which CMD, RUN, ADD references
WORKDIR /opt/app

# now copy all the files in this directory to /opt/app directory
ADD . .

RUN /opt/app/installation.sh

CMD ["./jupyter.sh"]

EXPOSE 8888

#RUN echo 'PS1="condauser: \W \$ "' >> ~/.bashrc
