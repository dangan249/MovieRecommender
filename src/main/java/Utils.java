import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Created by andang on 4/1/16.
 */
public class Utils {
	public static void saveStringToFile(String str, String filePath) {
		FileOutputStream fop;
		try {
			File file = new File(filePath);
			fop = new FileOutputStream(file);
			file.createNewFile();

			// get the content in bytes
			byte[] contentInBytes = str.getBytes();

			fop.write(contentInBytes);
			fop.flush();
			fop.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
