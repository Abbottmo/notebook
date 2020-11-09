#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <string.h>
#include <ctype.h>

#define SERV_PORT 6666

int main() {
	struct sockaddr_in servaddr;
	int sockfd, n;
	char buf[BUFSIZ];

	sockfd = socket(AF_INET, SOCK_DGRAM, 0);

	bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	inet_pton(AF_INET, "127.0.0.1", &servaddr.sin_addr);
	servaddr.sin_port = htons(SERV_PORT);

	while (fgets(buf, BUFSIZ, stdin) != NULL) {
		n = sendto(sockfd, buf, strlen(buf), 0, (struct sockaddr *)&servaddr, sizeof(servaddr));
		if (n == -1) {
			perror("sendto error");
		}

		n = recvfrom(sockfd, buf, BUFSIZ, 0, NULL, 0);	// NULL 不关心对端信息
		if (n == -1) {
			perror("recvfrom error");
		}

		write(STDOUT_FILENO, buf, n);
	}
	return 0;
}
