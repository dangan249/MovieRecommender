import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class MovieRatingRecommender {
	private String trainDataFile;
	private int numLatentFeature;
	private int numUsers;
	private int numItems;

	private int[][] data;
	private double[][] usersLatentFeatureVector;
	private double[][] itemsLatentFeatureVector;
	double LEARNING_RATE = 0.01;
	double MAX_NUM_ITERATION = 5000;
	double ACCEPTED_SQUARE_ERROR = 1;
	double LAMBDA = 0.01;

	public void train() throws IOException {
		loadTrainingData();
		initializeLatentFeatureVectors();
		trainMatrixFactorization();
		printData();
		printResult();
	}

	private void trainMatrixFactorization() {
		int numIteration = 0;

		while(true) {
//			System.out.println(numIteration);

			for(int i = 0; i < numUsers; i++) {
				for(int j = 0; j < numItems; j++) {
					if(data[i][j] > 0) {
						updateUserLatentFeature(i, j);
						double error = calculateSquareError();
						System.out.println(error);
						if(error < ACCEPTED_SQUARE_ERROR) break;
					}
				}
			}

			if(numIteration == MAX_NUM_ITERATION) {
				System.out.println("num iteration exceded");
				break;
			}
			numIteration++;
		}
	}

	private double calculateSquareError() {
		double error = 0;
		for(int i = 0; i < numUsers; i++) {
			for (int j = 0; j < numItems; j++) {
				if(data[i][j] > 0) {
					double rating = data[i][j];
					double predictedRating = calculatePredictedRating(usersLatentFeatureVector[i], itemsLatentFeatureVector[j]);
					error += Math.pow(rating - predictedRating, 2) + LAMBDA * (calculateVectorLengthSqr(usersLatentFeatureVector[i]) + calculateVectorLengthSqr(itemsLatentFeatureVector[i]));
				}
			}
		}
		return error;
	}

	private double calculateVectorLengthSqr(double[] vector) {
		double len = 0;
		for(int i = 0; i < vector.length; i++) {
			len += Math.pow(vector[i], 2);
		}
		return len;
	}

	private void updateUserLatentFeature(int userIndex, int itemIndex) {
		int currentRating = data[userIndex][itemIndex];
		double[] currentUserLatentFeatureVector = usersLatentFeatureVector[userIndex];
		double[] currentItemLatentFeatureVector = itemsLatentFeatureVector[itemIndex];
		double newRating = calculatePredictedRating(currentUserLatentFeatureVector, currentItemLatentFeatureVector);
		double[] newUserLatentFeatureVector = new double[numLatentFeature];
		double[] newItemLatentFeatureVector = new double[numLatentFeature];

		for(int i = 0; i < numLatentFeature; i++) {
			newUserLatentFeatureVector[i] = currentUserLatentFeatureVector[i] + LEARNING_RATE * ((currentRating - newRating) * currentItemLatentFeatureVector[i] - LAMBDA * currentUserLatentFeatureVector[i]);
			newItemLatentFeatureVector[i] = currentItemLatentFeatureVector[i] + LEARNING_RATE * ((currentRating - newRating) * currentUserLatentFeatureVector[i] - LAMBDA * currentItemLatentFeatureVector[i]);
		}

		for(int i = 0; i < numLatentFeature; i++) {
			currentUserLatentFeatureVector[i] = newUserLatentFeatureVector[i];
			currentItemLatentFeatureVector[i] = newItemLatentFeatureVector[i];
		}
	}

	// calculate newRating by multiple two current latent feature vectors
	private double calculatePredictedRating(double[] userLatentFeatureVector, double[] itemLatentFeatureVector) {
		double rating = 0;
		for(int i = 0; i < numLatentFeature; i++) {
			rating += userLatentFeatureVector[i] * itemLatentFeatureVector[i];
		}

		return rating;
	}

	private void initializeLatentFeatureVectors() {
		for(int i = 0; i < numUsers; i++) {
			for(int j = 0; j < numLatentFeature; j++) {
				usersLatentFeatureVector[i][j] = Math.random();
			}
		}


		for(int i = 0; i < numItems; i++) {
			for(int j = 0; j < numLatentFeature; j++) {
				itemsLatentFeatureVector[i][j] = Math.random();
			}
		}
	}

	private void loadTrainingData() throws IOException {
		String line;
		BufferedReader br = new BufferedReader(new FileReader(trainDataFile));
		while ((line = br.readLine()) != null) {
			String[] dataItem = line.split("\\s+");
			int user = Integer.parseInt(dataItem[0]);
			int item = Integer.parseInt(dataItem[1]);
			int rating = Integer.parseInt(dataItem[2]);

			if (1 <= user && user <= 10 && 1 <= item && item <= 10) {
				data[user - 1][item - 1] = rating;
			}
		}
	}

	private void printData() {
		System.out.println("DATA");
		for (int i = 0; i < numUsers; i++) {
			StringBuilder stringBuilder = new StringBuilder();
			stringBuilder.append("User " + (i + 1) + ":    ");
			for (int j = 0; j < numItems; j++) {
				if(data[i][j] > 0) {
					stringBuilder.append(data[i][j]);
				} else {
					stringBuilder.append("?");
				}
				stringBuilder.append("    ");
			}
			System.out.println(stringBuilder.toString());
		}
	}

	private void printResult() {
		System.out.println("RESULT");
		for (int i = 0; i < numUsers; i++) {
			StringBuilder stringBuilder = new StringBuilder();
			stringBuilder.append("User " + (i + 1) + ":    ");
			for (int j = 0; j < numItems; j++) {
				double predictedRating = calculatePredictedRating(usersLatentFeatureVector[i], itemsLatentFeatureVector[j]);
				stringBuilder.append(predictedRating);
				stringBuilder.append("    ");
			}
			System.out.println(stringBuilder.toString());
		}
	}

	public MovieRatingRecommender(String trainDataFile, int numLatentFeature, int numUsers, int numItems) {
		this.trainDataFile = trainDataFile;
		this.numLatentFeature = numLatentFeature;
		this.data = new int[numUsers][numItems];
		this.numUsers = numUsers;
		this.numItems = numItems;
		this.usersLatentFeatureVector = new double[numUsers][numLatentFeature];
		this.itemsLatentFeatureVector = new double[numItems][numLatentFeature];
	}

	public static void main(String[] args) {
		int numLatenFeature = 2;
		int numUsers = 10; //943;
		int numItems = 10;//1682;
		MovieRatingRecommender recommender = new MovieRatingRecommender("u.data", numLatenFeature, numUsers, numItems);
		try {
			recommender.train();
		} catch (IOException e) {
			System.out.println("invalid file");
		}
	}
}
