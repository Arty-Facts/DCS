import docker
from docker.errors import NotFound
import pathlib
import os
import stat
import glob

# Configuration variables
mysql_root_password = 'root'
mysql_password = 'dcs'
mysql_database = 'dcs'
mysql_conf_dir = 'd:/mysql/dcs/conf'  # Adjust for Linux if necessary
mysql_data_dir = 'd:/mysql/dcs/db'  # Adjust for Linux if necessary
source_conf_content = """
[mysqld]
max_connections=100000
thread_cache_size=1000
innodb_buffer_pool_size=16G
innodb-buffer-pool-instances=32
innodb_thread_concurrency = 0
query_cache_size = 64M
innodb_log_file_size = 256M
"""
dest_conf_file_path = pathlib.Path(mysql_conf_dir) / 'my.cnf'
container_name = f'mysql_{mysql_database}'
image_name = 'mysql:latest'

# Initialize Docker client
client = docker.from_env()

def stop_and_remove_container(name):
    try:
        container = client.containers.get(name)
        print(f"Stopping container {name}...")
        container.stop()  # Gracefully stops the container
        container.remove()
        print(f"Removed container {name}.")
    except NotFound:
        print(f"No existing container named {name} to stop.")

def main():
    stop_and_remove_container(container_name)

    # Ensure the configuration directory exists
    dest_conf_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the configuration file
    with open(dest_conf_file_path, 'w') as conf_file:
        conf_file.write(source_conf_content)
    
    # Change permissions of the configuration file (Linux)
    # For Windows, permissions handling might need different handling
    os.chmod(dest_conf_file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    for file in glob.glob(f"{mysql_data_dir}/**/*", recursive=True):
        os.chmod(file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        
    # Ensure the data directory exists
    pathlib.Path(mysql_data_dir).mkdir(parents=True, exist_ok=True)

    # Start the MySQL container
    print("Starting MySQL Docker container...")
    container = client.containers.run(
        image_name,
        detach=True,
        name=container_name,
        volumes={
            str(mysql_conf_dir): {'bind': '/etc/mysql/conf.d', 'mode': 'rw'},
            str(mysql_data_dir): {'bind': '/var/lib/mysql', 'mode': 'rw'},
        },
        ports={'3306/tcp': 3306},
        environment={
            'MYSQL_ROOT_PASSWORD': mysql_root_password,
            'MYSQL_PASSWORD': mysql_password,
            'MYSQL_DATABASE': mysql_database,
        }
    )

    print("MySQL Docker container started successfully.")

if __name__ == '__main__':
    main()