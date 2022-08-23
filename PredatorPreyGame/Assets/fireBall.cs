using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class fireBall : MonoBehaviour
{
    public float speed = 1f;
    public int damage = 1;
    public Rigidbody2D rb;
    private Transform player;
    private Vector2 target;

    // Start is called before the first frame update
    void Start() // MAKE TO ON COLLISION ENTER
    {
        player = GameObject.FindGameObjectWithTag("Player").transform;
        target = new Vector2(player.position.x, player.position.y);
    }

    private void Update()
    {
        transform.position = Vector2.MoveTowards(transform.position, target, speed * Time.deltaTime); 
        // Destroy particle when it reaches player's initial position.
        if (transform.position.x == target.x && transform.position.y == target.y)
        {
            Destroy(gameObject);
        }
    }

    private void OnTriggerEnter2D(Collider2D collision)
    {
        playerHealth player = collision.GetComponent<playerHealth>();
        if (player != null)
        {
            player.TakeDamage(damage);
            Destroy(gameObject);
        }
    }
    
        
   

}
